"""YOLOv5 检测 / 实例分割训练 / 推理用数据变换。

提供若干自包含 transform(不依赖 mmdet):

- ``LoadLabelmeAnnotations``: 把 dataset 的 ``instances`` 字段转为
  ``gt_bboxes`` / ``gt_bboxes_labels`` / ``gt_polygons`` 数组。
- ``LetterResize``: YOLOv5 等比缩放 + 居中 padding 到正方形,同步缩放 bbox/polygon。
- ``YOLOv5HSVRandomAug``: HSV 色彩抖动(仅作用于图像)。
- ``RandomFlip``: 随机水平 / 垂直翻转,同步翻转 bbox/polygon。
- ``RandomAffine``: 随机仿射(平移 / 缩放 / 旋转 / 错切),同步变换 bbox/polygon。
- ``PackDetInputs``: 打包成 ``inputs`` 张量 + ``data_samples``(BaseDataElement)。

几何增强约定:统一变换 ``gt_polygons`` 点集与 ``gt_bboxes``,实例 mask 由
``PackDetInputs`` 最后从 polygon 光栅化,因此无需 warp 整张 mask——既快又
能保证 bbox / polygon / mask 三者完全一致。推荐 pipeline 顺序::

    LoadImageFromFile → LoadLabelmeAnnotations → YOLOv5HSVRandomAug
    → LetterResize → RandomAffine → RandomFlip → PackDetInputs
"""
import math

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

from mmengine.dataset import Compose
from mmengine.structures import BaseDataElement, InstanceData

from mmaivision.registry import TRANSFORMS


def _filter_instances(results: dict, keep: np.ndarray) -> dict:
    """按布尔索引 ``keep`` 同步过滤 bbox / label / ignore / polygon。

    几何增强后部分实例可能被移出画面或退化,需要对所有 gt 字段一致删减。
    """
    keep = np.asarray(keep, dtype=bool)
    if 'gt_bboxes' in results:
        results['gt_bboxes'] = results['gt_bboxes'][keep]
    if 'gt_bboxes_labels' in results:
        results['gt_bboxes_labels'] = results['gt_bboxes_labels'][keep]
    if 'gt_ignore_flags' in results:
        results['gt_ignore_flags'] = results['gt_ignore_flags'][keep]
    if 'gt_polygons' in results:
        results['gt_polygons'] = [
            p for p, k in zip(results['gt_polygons'], keep) if k]
    return results


def _bbox_from_polygon(poly: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """有 polygon 时用其外接框(更紧),否则回退到已变换的 bbox。"""
    if len(poly) >= 3:
        x1, y1 = poly[:, 0].min(), poly[:, 1].min()
        x2, y2 = poly[:, 0].max(), poly[:, 1].max()
        return np.array([x1, y1, x2, y2], dtype=np.float32)
    return bbox.astype(np.float32)


@TRANSFORMS.register_module()
class LoadLabelmeAnnotations(BaseTransform):
    """从 dataset 的 ``instances`` 列表构造 gt 数组。

    输出:
        - ``gt_bboxes``: ``(N, 4)`` float32, xyxy 像素。
        - ``gt_bboxes_labels``: ``(N,)`` int64。
        - ``gt_ignore_flags``: ``(N,)`` bool。
        - ``gt_polygons``: ``list`` of ``(P, 2)`` float32 多边形点;无 polygon 的实例为 ``(0, 2)``。
    """

    def __init__(self, box_as_mask: bool = False):
        self.box_as_mask = box_as_mask

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
            elif self.box_as_mask:
                x1, y1, x2, y2 = inst['bbox']
                pts = np.array(
                    [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                    dtype=np.float32)
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
class YOLOv5HSVRandomAug(BaseTransform):
    """YOLOv5 风格 HSV 色彩抖动(仅作用于图像,不改 bbox/polygon)。

    对色相 / 饱和度 / 明度施加随机增益,提升对光照、色温变化的鲁棒性。
    输入图像须为 BGR uint8(本项目 ``LoadImageFromFile`` 默认输出)。

    Args:
        hue_delta: 色相增益幅度(YOLOv5 默认 0.015)。
        saturation_delta: 饱和度增益幅度(默认 0.7)。
        value_delta: 明度增益幅度(默认 0.4)。
    """

    def __init__(self,
                 hue_delta: float = 0.015,
                 saturation_delta: float = 0.7,
                 value_delta: float = 0.4):
        for name, v in (('hue_delta', hue_delta),
                        ('saturation_delta', saturation_delta),
                        ('value_delta', value_delta)):
            if v < 0:
                raise ValueError(f'{name} 必须 >= 0, got {v}')
        self.hue_delta = hue_delta
        self.saturation_delta = saturation_delta
        self.value_delta = value_delta

    def transform(self, results: dict) -> dict:
        img = results['img']
        if img.dtype != np.uint8:
            raise ValueError(
                f'YOLOv5HSVRandomAug 需要 uint8 BGR 图像, got {img.dtype}')
        # 每通道独立随机增益 ∈ [1-delta, 1+delta]
        gains = np.random.uniform(-1, 1, 3) * [
            self.hue_delta, self.saturation_delta, self.value_delta] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        dtype = img.dtype
        x = np.arange(0, 256, dtype=gains.dtype)
        lut_hue = ((x * gains[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * gains[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * gains[2], 0, 255).astype(dtype)
        img_hsv = cv2.merge(
            (cv2.LUT(hue, lut_hue),
             cv2.LUT(sat, lut_sat),
             cv2.LUT(val, lut_val)))
        results['img'] = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        return results


@TRANSFORMS.register_module()
class RandomFlip(BaseTransform):
    """随机水平 / 垂直翻转,同步翻转 bbox 与 polygon。

    Args:
        prob: 水平翻转概率。
        vertical_prob: 垂直翻转概率(默认 0,关闭)。
    """

    def __init__(self, prob: float = 0.5, vertical_prob: float = 0.0):
        if not 0 <= prob <= 1:
            raise ValueError(f'prob 必须 ∈ [0, 1], got {prob}')
        if not 0 <= vertical_prob <= 1:
            raise ValueError(
                f'vertical_prob 必须 ∈ [0, 1], got {vertical_prob}')
        self.prob = prob
        self.vertical_prob = vertical_prob

    def transform(self, results: dict) -> dict:
        img = results['img']
        h, w = img.shape[:2]
        if np.random.rand() < self.prob:
            results['img'] = np.ascontiguousarray(img[:, ::-1])
            img = results['img']
            self._flip_coords(results, axis=0, size=w)
        if np.random.rand() < self.vertical_prob:
            results['img'] = np.ascontiguousarray(img[::-1])
            self._flip_coords(results, axis=1, size=h)
        return results

    @staticmethod
    def _flip_coords(results: dict, axis: int, size: int) -> None:
        """axis=0 水平(翻 x),axis=1 垂直(翻 y)。"""
        if 'gt_bboxes' in results and len(results['gt_bboxes']):
            b = results['gt_bboxes'].copy()
            lo = axis * 1      # x 用列 0/2,y 用列 1/3
            c1, c2 = (0, 2) if axis == 0 else (1, 3)
            flipped = size - b[:, [c2, c1]]   # 交换并镜像,保证 x1<x2
            b[:, c1], b[:, c2] = flipped[:, 0], flipped[:, 1]
            results['gt_bboxes'] = b
        if 'gt_polygons' in results:
            col = 0 if axis == 0 else 1
            for p in results['gt_polygons']:
                if len(p):
                    p[:, col] = size - p[:, col]


@TRANSFORMS.register_module()
class RandomAffine(BaseTransform):
    """YOLOv5 风格随机仿射:平移 / 缩放 / 旋转 / 错切,同步变换 bbox 与 polygon。

    在 letterbox(正方形、114 padding)之后使用最自然:新生边缘用 ``border_val``
    填充以与 letterbox 一致。变换后用 ``box_candidates`` 过滤掉过小 / 长宽比畸变 /
    面积锐减的实例(对应 YOLOv5 的退化框剔除)。默认仅平移 + 缩放(``degrees=0``,
    ``shear=0``),与 YOLOv5 默认一致,适合无显著旋转的工业场景。

    ``border`` 用于把大画布裁回目标尺寸,典型用法是接在 ``Mosaic`` 之后:Mosaic
    产出 ``2S×2S`` 拼图,``RandomAffine(border=(-S//2, -S//2))`` 在做仿射的同时把
    输出裁回 ``S×S``(与 YOLOv5 mosaic→random_perspective 流程一致)。

    Args:
        max_rotate_degree: 旋转角度范围 ±degrees。
        max_translate_ratio: 平移比例(相对边长)±ratio。
        scaling_ratio_range: 缩放系数范围 ``(min, max)``。
        max_shear_degree: 错切角度范围 ±degrees。
        border_val: 边缘填充值(应与 LetterResize 的 pad_val 一致,默认 114)。
        border: ``(border_y, border_x)`` 输出尺寸偏移,输出
            ``(h + 2*border_y, w + 2*border_x)``;默认 ``(0, 0)`` 即不改尺寸。
        min_bbox_size: 变换后 bbox 最小边长(像素),小于则剔除。
        min_area_ratio: 变换后 / 变换前面积比下限,低于则剔除。
        max_aspect_ratio: 变换后长宽比上限,超过则剔除。
    """

    def __init__(self,
                 max_rotate_degree: float = 0.0,
                 max_translate_ratio: float = 0.1,
                 scaling_ratio_range: tuple = (0.5, 1.5),
                 max_shear_degree: float = 0.0,
                 border_val: int = 114,
                 border: tuple = (0, 0),
                 min_bbox_size: float = 2.0,
                 min_area_ratio: float = 0.1,
                 max_aspect_ratio: float = 20.0):
        if max_translate_ratio < 0:
            raise ValueError(
                f'max_translate_ratio 必须 >= 0, got {max_translate_ratio}')
        lo, hi = scaling_ratio_range
        if not 0 < lo <= hi:
            raise ValueError(
                f'scaling_ratio_range 必须满足 0 < min <= max, '
                f'got {scaling_ratio_range}')
        self.max_rotate_degree = max_rotate_degree
        self.max_translate_ratio = max_translate_ratio
        self.scaling_ratio_range = scaling_ratio_range
        self.max_shear_degree = max_shear_degree
        self.border_val = border_val
        self.border = border
        self.min_bbox_size = min_bbox_size
        self.min_area_ratio = min_area_ratio
        self.max_aspect_ratio = max_aspect_ratio

    def _get_matrix(self, h: int, w: int,
                    out_h: int, out_w: int) -> np.ndarray:
        """生成 3x3 仿射矩阵:中心移到原点 → 旋转+缩放 → 错切 → 平移到输出画布。"""
        # 以输入图中心为原点
        center = np.eye(3, dtype=np.float32)
        center[0, 2] = -w / 2
        center[1, 2] = -h / 2
        # 旋转 + 缩放
        angle = np.random.uniform(-self.max_rotate_degree,
                                   self.max_rotate_degree)
        scale = np.random.uniform(*self.scaling_ratio_range)
        rot = np.eye(3, dtype=np.float32)
        rot[:2] = cv2.getRotationMatrix2D(
            angle=angle, center=(0, 0), scale=scale)
        # 错切
        shear = np.eye(3, dtype=np.float32)
        shear[0, 1] = math.tan(
            np.random.uniform(-self.max_shear_degree, self.max_shear_degree)
            * math.pi / 180)
        shear[1, 0] = math.tan(
            np.random.uniform(-self.max_shear_degree, self.max_shear_degree)
            * math.pi / 180)
        # 平移到输出画布中心 + 随机偏移(相对输出尺寸)
        trans = np.eye(3, dtype=np.float32)
        trans[0, 2] = np.random.uniform(
            0.5 - self.max_translate_ratio,
            0.5 + self.max_translate_ratio) * out_w
        trans[1, 2] = np.random.uniform(
            0.5 - self.max_translate_ratio,
            0.5 + self.max_translate_ratio) * out_h
        return trans @ shear @ rot @ center

    @staticmethod
    def _affine_points(pts: np.ndarray, m: np.ndarray) -> np.ndarray:
        """对 (P,2) 点集应用 3x3 仿射矩阵(取前两行)。"""
        return pts @ m[:2, :2].T + m[:2, 2]

    def _box_candidates(self, box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
        """YOLOv5 退化框过滤:边长 / 面积比 / 长宽比三重条件。"""
        w1, h1 = box1[:, 2] - box1[:, 0], box1[:, 3] - box1[:, 1]
        w2, h2 = box2[:, 2] - box2[:, 0], box2[:, 3] - box2[:, 1]
        ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))
        return ((w2 > self.min_bbox_size)
                & (h2 > self.min_bbox_size)
                & (w2 * h2 / (w1 * h1 + 1e-16) > self.min_area_ratio)
                & (ar < self.max_aspect_ratio))

    def transform(self, results: dict) -> dict:
        img = results['img']
        h, w = img.shape[:2]
        out_h = h + 2 * self.border[0]
        out_w = w + 2 * self.border[1]
        m = self._get_matrix(h, w, out_h, out_w)

        border = (self.border_val, ) * (
            img.shape[2] if img.ndim == 3 else 1)
        results['img'] = cv2.warpAffine(
            img, m[:2], (out_w, out_h), borderValue=border)
        results['img_shape'] = results['img'].shape[:2]

        n = len(results.get('gt_bboxes', []))
        if n == 0:
            return results

        old_boxes = results['gt_bboxes'].astype(np.float32)
        polys = results.get('gt_polygons', [None] * n)
        new_boxes = np.zeros_like(old_boxes)
        new_polys = []
        for i in range(n):
            poly = polys[i] if polys[i] is not None else np.zeros(
                (0, 2), np.float32)
            if len(poly):
                tp = self._affine_points(poly.astype(np.float32), m)
                tp[:, 0] = tp[:, 0].clip(0, out_w)
                tp[:, 1] = tp[:, 1].clip(0, out_h)
                new_polys.append(tp)
                new_boxes[i] = _bbox_from_polygon(tp, old_boxes[i])
            else:
                # 无 polygon:变换 bbox 四角再取外接框
                x1, y1, x2, y2 = old_boxes[i]
                corners = np.array(
                    [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
                tc = self._affine_points(corners, m)
                new_boxes[i] = [
                    tc[:, 0].min(), tc[:, 1].min(),
                    tc[:, 0].max(), tc[:, 1].max()]
                new_polys.append(poly)
        new_boxes[:, 0::2] = new_boxes[:, 0::2].clip(0, out_w)
        new_boxes[:, 1::2] = new_boxes[:, 1::2].clip(0, out_h)

        keep = self._box_candidates(old_boxes, new_boxes)
        results['gt_bboxes'] = new_boxes
        if 'gt_polygons' in results:
            results['gt_polygons'] = new_polys
        _filter_instances(results, keep)
        return results


class _MultiImageMixin(BaseTransform):
    """需要额外样本的增强(Mosaic / MixUp)基类。

    依赖 dataset 把自身注入 ``results['dataset']``(见
    ``LabelmeDetDataset.prepare_data``)。子类用 ``self._load_extra(dataset, idx)``
    走内部 ``pre_transform``(仅做加载,不含 mix 增强,避免递归)加载其他样本。

    Args:
        pre_transform: 加载其他样本用的 transform 配置列表,通常为
            ``[dict(type='LoadImageFromFile'), dict(type='LoadLabelmeAnnotations')]``。
        prob: 应用本增强的概率,否则原样透传。
    """

    def __init__(self, pre_transform=None, prob: float = 1.0):
        if not 0 <= prob <= 1:
            raise ValueError(f'prob 必须 ∈ [0, 1], got {prob}')
        self.prob = prob
        self.pre_transform = Compose(pre_transform) if pre_transform else None

    def _load_extra(self, dataset, idx: int) -> dict:
        data = dataset.get_data_info(idx)
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        return data

    @staticmethod
    def _concat(results: dict, parts: list) -> None:
        """把多份样本的 gt 字段拼接进 results(bbox/label/ignore/polygon)。"""
        bboxes = [p['gt_bboxes'] for p in parts if len(p.get('gt_bboxes', []))]
        labels = [p['gt_bboxes_labels'] for p in parts
                  if len(p.get('gt_bboxes_labels', []))]
        ignores = [p['gt_ignore_flags'] for p in parts
                   if len(p.get('gt_ignore_flags', []))]
        polys = []
        for p in parts:
            polys.extend(p.get('gt_polygons', []))
        results['gt_bboxes'] = (
            np.concatenate(bboxes, 0) if bboxes
            else np.zeros((0, 4), np.float32))
        results['gt_bboxes_labels'] = (
            np.concatenate(labels, 0) if labels
            else np.zeros((0, ), np.int64))
        results['gt_ignore_flags'] = (
            np.concatenate(ignores, 0) if ignores
            else np.zeros((0, ), bool))
        results['gt_polygons'] = polys


@TRANSFORMS.register_module()
class Mosaic(_MultiImageMixin):
    """YOLOv5 风格 4 图马赛克拼接,产出 ``2*img_scale`` 画布。

    随机取另外 3 张图,各按等比缩放(长边=img_scale)贴入 2S×2S 画布的四个象限,
    拼接中心在中央区域随机抖动。bbox / polygon 按各自的缩放比与贴图偏移变换并
    裁剪到画布。**通常紧跟 ``RandomAffine(border=(-img_scale//2,)*2)``** 做仿射并裁回
    ``img_scale``,与 YOLOv5 mosaic→random_perspective 流程一致。

    需在 ``LetterResize`` *之前* 使用(直接吃原图),且 dataset 须注入
    ``results['dataset']``。

    Args:
        img_scale: 单图目标边长 S;输出画布为 2S×2S。
        pre_transform: 加载其他 3 张图的 transform 列表(仅加载,不含 mix)。
        pad_val: 画布填充值(默认 114,与 letterbox 一致)。
        prob: 应用概率。
    """

    def __init__(self,
                 img_scale: int = 640,
                 pre_transform=None,
                 pad_val: int = 114,
                 prob: float = 1.0):
        super().__init__(pre_transform=pre_transform, prob=prob)
        if img_scale <= 0:
            raise ValueError(f'img_scale 必须 > 0, got {img_scale}')
        self.img_scale = img_scale
        self.pad_val = pad_val

    def transform(self, results: dict) -> dict:
        if np.random.rand() > self.prob:
            return results
        dataset = results.get('dataset')
        if dataset is None:
            raise RuntimeError(
                'Mosaic 需要 results["dataset"];请确认 dataset.prepare_data '
                '已注入 dataset 引用,且本 transform 在训练 pipeline 中。')
        # 当前样本 + 随机 3 张
        idxs = [np.random.randint(0, len(dataset)) for _ in range(3)]
        mosaic_samples = [results] + [self._load_extra(dataset, i)
                                      for i in idxs]

        s = self.img_scale
        # 2S 画布;拼接中心在 [0.5S, 1.5S] 随机
        canvas = np.full((s * 2, s * 2, 3), self.pad_val, dtype=np.uint8)
        xc = int(np.random.uniform(s * 0.5, s * 1.5))
        yc = int(np.random.uniform(s * 0.5, s * 1.5))

        parts = []
        for pos, sample in enumerate(mosaic_samples):
            img = sample['img']
            h, w = img.shape[:2]
            r = s / max(h, w)               # 等比缩放,长边=S
            if r != 1:
                img = cv2.resize(
                    img, (int(w * r), int(h * r)),
                    interpolation=cv2.INTER_LINEAR)
            nh, nw = img.shape[:2]
            # 各象限:画布粘贴区 (x1a,y1a,x2a,y2a) 与源图裁剪区 (x1b,y1b,x2b,y2b)
            if pos == 0:        # 左上
                x1a, y1a, x2a, y2a = max(xc - nw, 0), max(yc - nh, 0), xc, yc
                x1b, y1b, x2b, y2b = nw - (x2a - x1a), nh - (y2a - y1a), nw, nh
            elif pos == 1:      # 右上
                x1a, y1a, x2a, y2a = xc, max(yc - nh, 0), min(xc + nw, s * 2), yc
                x1b, y1b, x2b, y2b = 0, nh - (y2a - y1a), x2a - x1a, nh
            elif pos == 2:      # 左下
                x1a, y1a, x2a, y2a = max(xc - nw, 0), yc, xc, min(s * 2, yc + nh)
                x1b, y1b, x2b, y2b = nw - (x2a - x1a), 0, nw, y2a - y1a
            else:               # 右下
                x1a, y1a, x2a, y2a = xc, yc, min(xc + nw, s * 2), min(s * 2, yc + nh)
                x1b, y1b, x2b, y2b = 0, 0, x2a - x1a, y2a - y1a
            canvas[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            # 坐标偏移:源图(缩放后)坐标 → 画布坐标
            padw, padh = x1a - x1b, y1a - y1b
            parts.append(self._shift_sample(sample, r, padw, padh))

        out = dict(results)
        out['img'] = canvas
        out['img_shape'] = canvas.shape[:2]
        self._concat(out, parts)
        # 拼接后整体裁剪到画布范围
        self._clip_to_canvas(out, s * 2, s * 2)
        return out

    @staticmethod
    def _shift_sample(sample: dict, r: float, padw: int, padh: int) -> dict:
        """对单张源样本的 bbox/polygon 按缩放 r + 偏移 (padw,padh) 变换。"""
        n = len(sample.get('gt_bboxes', []))
        b = sample['gt_bboxes'].astype(np.float32).copy() * r if n else \
            np.zeros((0, 4), np.float32)
        if n:
            b[:, 0::2] += padw
            b[:, 1::2] += padh
        polys = []
        for p in sample.get('gt_polygons', []):
            if len(p):
                p = p.astype(np.float32).copy() * r
                p[:, 0] += padw
                p[:, 1] += padh
            polys.append(p)
        return dict(
            gt_bboxes=b,
            gt_bboxes_labels=sample.get(
                'gt_bboxes_labels', np.zeros((0, ), np.int64)),
            gt_ignore_flags=sample.get(
                'gt_ignore_flags', np.zeros((0, ), bool)),
            gt_polygons=polys)

    @staticmethod
    def _clip_to_canvas(results: dict, h: int, w: int) -> None:
        b = results['gt_bboxes']
        if len(b):
            b[:, 0::2] = b[:, 0::2].clip(0, w)
            b[:, 1::2] = b[:, 1::2].clip(0, h)
        for p in results['gt_polygons']:
            if len(p):
                p[:, 0] = p[:, 0].clip(0, w)
                p[:, 1] = p[:, 1].clip(0, h)


@TRANSFORMS.register_module()
class MixUp(_MultiImageMixin):
    """YOLOv5 风格 MixUp:与另一张图按 Beta 权重线性混合,gt 取并集。

    取一张额外样本,letterbox 到与当前图相同尺寸后按系数 ``λ~Beta(α,α)`` 融合像素,
    两图的 bbox/polygon 直接合并。须在两图尺寸已统一之后用(如 Mosaic+Affine 之后,
    或 LetterResize 之后)。dataset 须注入 ``results['dataset']``。

    Args:
        alpha: Beta 分布参数(默认 32,对应 YOLOv5,λ 接近 0.5)。
        pre_transform: 加载额外样本的 transform 列表(仅加载,不含 mix)。
        pad_val: letterbox 对齐尺寸时的填充值。
        prob: 应用概率。
    """

    def __init__(self,
                 alpha: float = 32.0,
                 pre_transform=None,
                 pad_val: int = 114,
                 prob: float = 1.0):
        super().__init__(pre_transform=pre_transform, prob=prob)
        if alpha <= 0:
            raise ValueError(f'alpha 必须 > 0, got {alpha}')
        self.alpha = alpha
        self.pad_val = pad_val

    def transform(self, results: dict) -> dict:
        if np.random.rand() > self.prob:
            return results
        dataset = results.get('dataset')
        if dataset is None:
            raise RuntimeError(
                'MixUp 需要 results["dataset"];请确认 dataset.prepare_data '
                '已注入 dataset 引用。')
        other = self._load_extra(dataset, np.random.randint(0, len(dataset)))
        other = self._resize_to(other, results['img'].shape[:2])

        lam = float(np.random.beta(self.alpha, self.alpha))
        mixed = (results['img'].astype(np.float32) * lam
                 + other['img'].astype(np.float32) * (1 - lam))
        out = dict(results)
        out['img'] = mixed.astype(np.uint8)
        out['img_shape'] = out['img'].shape[:2]
        # 当前样本 gt 也需规整成 part 结构后与 other 合并
        cur = dict(
            gt_bboxes=results.get('gt_bboxes', np.zeros((0, 4), np.float32)),
            gt_bboxes_labels=results.get(
                'gt_bboxes_labels', np.zeros((0, ), np.int64)),
            gt_ignore_flags=results.get(
                'gt_ignore_flags', np.zeros((0, ), bool)),
            gt_polygons=results.get('gt_polygons', []))
        self._concat(out, [cur, other])
        return out

    def _resize_to(self, sample: dict, size: tuple) -> dict:
        """把额外样本 letterbox 到目标 (H, W),同步缩放 bbox/polygon。"""
        img = sample['img']
        h, w = img.shape[:2]
        H, W = size
        r = min(H / h, W / w)
        nh, nw = round(h * r), round(w * r)
        if (nw, nh) != (w, h):
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        top, left = (H - nh) // 2, (W - nw) // 2
        out_img = np.full((H, W, img.shape[2]), self.pad_val, dtype=img.dtype)
        out_img[top:top + nh, left:left + nw] = img
        sample['img'] = out_img
        if len(sample.get('gt_bboxes', [])):
            b = sample['gt_bboxes'].astype(np.float32).copy() * r
            b[:, 0::2] = (b[:, 0::2] + left).clip(0, W)
            b[:, 1::2] = (b[:, 1::2] + top).clip(0, H)
            sample['gt_bboxes'] = b
        polys = []
        for p in sample.get('gt_polygons', []):
            if len(p):
                p = p.astype(np.float32).copy() * r
                p[:, 0] = (p[:, 0] + left).clip(0, W)
                p[:, 1] = (p[:, 1] + top).clip(0, H)
            polys.append(p)
        sample['gt_polygons'] = polys
        return sample


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
        if 'gt_ignore_flags' in results:
            _filter_instances(results, ~results['gt_ignore_flags'])

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
