"""demo 脚本共享工具:模型构建、推理 pipeline、letterbox 坐标还原与可视化。

供 ``yolov5_inference.py`` / ``yolov5_seg_demo.py`` / ``visualize_augmentation.py``
以 ``from common import ...`` 方式复用(脚本直接 ``python demo/xxx.py`` 运行时
demo/ 即 sys.path[0])。
"""
import os
import os.path as osp

import cv2
import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import Compose, pseudo_collate
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint

import mmaivision  # noqa: F401  触发 registry 注册
from mmaivision.registry import MODELS

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
# 固定调色板(BGR),按类别索引取色。
PALETTE = [(0, 255, 0), (0, 128, 255), (255, 0, 0), (0, 0, 255),
           (255, 0, 255), (255, 255, 0), (128, 0, 255), (0, 215, 255)]


def collect_images(path):
    """目录 → 排序后的图片路径列表;单个文件原样返回。"""
    if osp.isdir(path):
        return sorted(
            osp.join(path, f) for f in os.listdir(path)
            if f.lower().endswith(IMG_EXTS))
    return [path]


def load_config(config_path):
    """读取 config 并初始化默认 scope;返回 (cfg, class_names)。"""
    cfg = Config.fromfile(config_path)
    init_default_scope(cfg.get('default_scope', 'mmaivision'))
    class_names = list(cfg.get('metainfo', {}).get('classes', ()))
    return cfg, class_names


def build_model(cfg, checkpoint, device):
    """按 config 构建模型并加载权重,移到 device 后置 eval。"""
    model = MODELS.build(cfg.model)
    load_checkpoint(model, checkpoint, map_location='cpu')
    return model.to(device).eval()


def build_pipeline(cfg):
    """测试时预处理(不含读图:推理时直接喂已解码的 BGR 图,避免重复读盘)。"""
    scale = cfg.get('img_scale', 640)
    return Compose([
        dict(type='LetterResize', scale=scale, pad_val=114),
        dict(type='PackDetInputs'),
    ])


def restore_boxes(boxes: np.ndarray, metainfo) -> np.ndarray:
    """letterbox(scale_factor / pad_param) 还原到原图坐标并裁剪。"""
    r = float(metainfo['scale_factor'][0])
    top, _, left, _ = metainfo['pad_param']
    h, w = metainfo['ori_shape']
    boxes = boxes.copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / r
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / r
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    return boxes


def restore_masks(masks: np.ndarray, metainfo) -> np.ndarray:
    """letterbox 输入分辨率的 mask(K,Hin,Win)→ 裁掉 padding → resize 回原图。"""
    h, w = metainfo['ori_shape']
    if masks.shape[0] == 0:
        return np.zeros((0, h, w), dtype=bool)
    top, bottom, left, right = [int(round(float(x)))
                                for x in metainfo['pad_param']]
    Hin, Win = masks.shape[1], masks.shape[2]
    out = np.zeros((masks.shape[0], h, w), dtype=bool)
    for i, m in enumerate(masks):
        crop = m[top:Hin - bottom, left:Win - right].astype(np.uint8)
        if crop.size == 0:
            continue
        resized = cv2.resize(crop, (w, h), interpolation=cv2.INTER_NEAREST)
        out[i] = resized.astype(bool)
    return out


def draw_instances(img, boxes, labels, class_names,
                   scores=None, masks=None, alpha=0.5):
    """bbox + 类别名(可选分数 / 实例 mask 半透明)叠加绘制。

    masks 为 None 时只画框(实线);否则把各实例 mask 画到 overlay 上按
    alpha 混合,masks 中允许个别条目为 None(无 mask 的实例)。
    """
    overlay = img.copy() if masks is not None else None
    for i in range(len(boxes)):
        c = int(labels[i])
        color = PALETTE[c % len(PALETTE)]
        if masks is not None:
            m = masks[i]
            if m is not None and m.any():
                overlay[m] = color
        x1, y1, x2, y2 = (int(v) for v in boxes[i][:4])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        name = class_names[c] if c < len(class_names) else str(c)
        text = f'{name} {float(scores[i]):.2f}' if scores is not None else name
        cv2.putText(img, text, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    if overlay is not None:
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    return img


@torch.no_grad()
def infer_one(model, pipeline, img, img_id, score_thr, with_masks=False):
    """单图推理并把结果还原回原图坐标。

    Args:
        img: 已解码的 BGR ndarray(直接喂 pipeline,避免重复读盘)。
        with_masks: True 时额外返回还原后的实例 mask。

    Returns:
        (boxes, scores, labels) 或 (boxes, scores, labels, masks)。
    """
    data = pipeline(dict(img=img, img_id=img_id, ori_shape=img.shape[:2]))
    sample = model.test_step(pseudo_collate([data]))[0]
    pred = sample.pred_instances
    scores = pred.scores.cpu().numpy()
    keep = scores >= score_thr
    boxes = restore_boxes(pred.bboxes.cpu().numpy()[keep], sample.metainfo)
    scores = scores[keep]
    labels = pred.labels.cpu().numpy()[keep]
    if not with_masks:
        return boxes, scores, labels
    masks = restore_masks(pred.masks.cpu().numpy()[keep], sample.metainfo)
    return boxes, scores, labels, masks
