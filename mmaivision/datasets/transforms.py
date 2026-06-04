"""YOLOv5 检测训练 / 推理用数据变换。

提供三个自包含 transform(不依赖 mmdet):

- ``LoadLabelmeAnnotations``: 把 dataset 的 ``instances`` 字段转为
  ``gt_bboxes`` / ``gt_bboxes_labels`` 数组。
- ``LetterResize``: YOLOv5 等比缩放 + 居中 padding 到正方形,同步缩放 bbox。
- ``PackDetInputs``: 打包成 ``inputs`` 张量 + ``data_samples``(BaseDataElement)。
"""
import cv2
import numpy as np
import torch

try:
    from mmcv.transforms import BaseTransform
except ImportError as e:
    import warnings
    warnings.warn(
        f'mmcv is not installed or cannot be loaded correctly: {e}\n'
        'Using `object` as the base class of the custom transform.')
    BaseTransform = object

from mmengine.structures import BaseDataElement, InstanceData

from mmaivision.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadLabelmeAnnotations(BaseTransform):
    """从 dataset 的 ``instances`` 列表构造 gt 数组。

    输出:
        - ``gt_bboxes``: ``(N, 4)`` float32, xyxy 像素。
        - ``gt_bboxes_labels``: ``(N,)`` int64。
        - ``gt_ignore_flags``: ``(N,)`` bool。
        - ``gt_polygons``: ``list`` of ``(P, 2)`` float32 多边形点;无 polygon 的实例为 ``(0, 2)``。
    """

    def transform(self, results: dict) -> dict:
        instances = results.get('instances', [])
        bboxes = np.array(
            [inst['bbox'] for inst in instances],
            dtype=np.float32).reshape(-1, 4)
        labels = np.array(
            [inst['bbox_label'] for inst in instances],
            dtype=np.int64).reshape(-1)
        ignore = np.array(
            [inst.get('ignore_flag', 0) for inst in instances],
            dtype=bool).reshape(-1)
        results['gt_bboxes'] = bboxes
        results['gt_bboxes_labels'] = labels
        results['gt_ignore_flags'] = ignore
        polygons = []
        for inst in instances:
            m = inst.get('mask')
            if m and len(m) > 0 and len(m[0]) >= 6:
                # 仅取首条轮廓 m[0],暂不支持带孔多边形(多轮廓)。
                pts = np.array(m[0], dtype=np.float32).reshape(-1, 2)
            else:
                pts = np.zeros((0, 2), dtype=np.float32)
            polygons.append(pts)
        results['gt_polygons'] = polygons
        return results


@TRANSFORMS.register_module()
class LetterResize(BaseTransform):
    """YOLOv5 letterbox:等比缩放 + 居中 padding 到 ``scale`` 正方形。

    Args:
        scale: 目标边长(正方形),默认 640。
        pad_val: padding 填充值,默认 114。
    """

    def __init__(self, scale: int = 640, pad_val: int = 114):
        if scale <= 0:
            raise ValueError(f'scale 必须 > 0, got {scale}')
        self.scale = scale
        self.pad_val = pad_val

    def transform(self, results: dict) -> dict:
        img = results['img']
        h, w = img.shape[:2]
        r = min(self.scale / h, self.scale / w)
        nh, nw = round(h * r), round(w * r)
        if (nw, nh) != (w, h):
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        top = (self.scale - nh) // 2
        left = (self.scale - nw) // 2
        out = np.full(
            (self.scale, self.scale, img.shape[2] if img.ndim == 3 else 1),
            self.pad_val, dtype=img.dtype)
        if img.ndim == 2:
            img = img[..., None]
        out[top:top + nh, left:left + nw] = img

        results['img'] = out
        results['img_shape'] = out.shape[:2]
        results['scale_factor'] = (r, r)
        # pad_param: [top, bottom, left, right],推理时还原坐标用
        results['pad_param'] = np.array(
            [top, self.scale - nh - top, left, self.scale - nw - left],
            dtype=np.float32)

        if 'gt_bboxes' in results and len(results['gt_bboxes']):
            b = results['gt_bboxes'].astype(np.float32).copy()
            b *= r
            b[:, 0::2] += left
            b[:, 1::2] += top
            b[:, 0::2] = b[:, 0::2].clip(0, self.scale)
            b[:, 1::2] = b[:, 1::2].clip(0, self.scale)
            results['gt_bboxes'] = b
        if 'gt_polygons' in results and results['gt_polygons']:
            new_polys = []
            for p in results['gt_polygons']:
                if len(p):
                    p = p.astype(np.float32).copy() * r
                    p[:, 0] = (p[:, 0] + left).clip(0, self.scale)
                    p[:, 1] = (p[:, 1] + top).clip(0, self.scale)
                new_polys.append(p)
            results['gt_polygons'] = new_polys
        return results


@TRANSFORMS.register_module()
class PackDetInputs(BaseTransform):
    """打包成 ``inputs`` (uint8 CHW 张量) + ``data_samples`` (BaseDataElement)。

    ``data_samples.gt_instances`` 含 ``bboxes`` (xyxy 像素) 与 ``labels``;
    当 ``gt_polygons`` 存在时,还包含 ``masks`` (二值实例 mask, ``(N, H, W)`` uint8),
    由 polygon 光栅化填充而来。
    metainfo 透传若干字段供推理坐标还原。归一化 / RGB 交由 data_preprocessor。
    """

    META_KEYS = ('img_id', 'img_path', 'ori_shape', 'img_shape',
                 'scale_factor', 'pad_param')

    def __init__(self, meta_keys=None):
        self.meta_keys = tuple(meta_keys) if meta_keys else self.META_KEYS

    def transform(self, results: dict) -> dict:
        img = results['img']
        if img.ndim == 2:
            img = img[..., None]
        img = np.ascontiguousarray(img.transpose(2, 0, 1))  # CHW(BGR)
        inputs = torch.from_numpy(img)

        gt_instances = InstanceData()
        if 'gt_bboxes' in results:
            gt_instances.bboxes = torch.from_numpy(
                np.ascontiguousarray(results['gt_bboxes'])).float()
            gt_instances.labels = torch.from_numpy(
                np.ascontiguousarray(results['gt_bboxes_labels'])).long()
        else:
            gt_instances.bboxes = torch.zeros((0, 4), dtype=torch.float32)
            gt_instances.labels = torch.zeros((0, ), dtype=torch.int64)

        if 'gt_polygons' in results:
            H, W = results['img_shape'][:2]
            polys = results['gt_polygons']
            masks = np.zeros((len(polys), H, W), dtype=np.uint8)
            for idx, p in enumerate(polys):
                if len(p) >= 3:
                    cv2.fillPoly(masks[idx], [p.round().astype(np.int32)], 1)
            gt_instances.masks = torch.from_numpy(masks)

        data_sample = BaseDataElement()
        data_sample.gt_instances = gt_instances
        meta = {k: results[k] for k in self.meta_keys if k in results}
        data_sample.set_metainfo(meta)
        return dict(inputs=inputs, data_samples=data_sample)
