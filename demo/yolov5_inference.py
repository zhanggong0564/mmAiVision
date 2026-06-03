"""YOLOv5 推理脚本(配置驱动,走模型内建 predict)。

与早期的 ``yolov5_detect_demo.py`` 不同:那份是 head 还没后处理时的临时方案
(手写 decode + NMS + 硬编码 COCO 80 类);本脚本直接用训练 config 构建模型、
加载 checkpoint,经 ``model.test_step`` 走内建 ``predict``(decode + NMS 已在
``YOLOv5Head.predict_by_feat`` 内),类别名从 config 的 metainfo 读取,并用
letterbox 的 scale_factor / pad_param 把框还原回原图坐标。

用法:
    # 单张图
    python demo/yolov5_inference.py <img> configs/yolov5_n_labelme.py \
        work_dirs/yolov5_n_labelme/best_labelme_mAP_50_epoch_100.pth \
        --out-dir work_dirs/infer --score-thr 0.3

    # 整个目录
    python demo/yolov5_inference.py <img_dir> <config> <checkpoint> \
        --out-dir work_dirs/infer
"""
import argparse
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


def parse_args():
    p = argparse.ArgumentParser(description='YOLOv5 配置驱动推理')
    p.add_argument('image', help='输入图片路径或目录')
    p.add_argument('config', help='训练 config 路径')
    p.add_argument('checkpoint', help='模型权重 .pth')
    p.add_argument('--out-dir', default='work_dirs/infer', help='结果输出目录')
    p.add_argument('--score-thr', type=float, default=0.3, help='可视化阈值')
    p.add_argument('--device', default=None, help='cuda:0 / cpu,默认自动')
    return p.parse_args()


def collect_images(path):
    if osp.isdir(path):
        return sorted(
            osp.join(path, f) for f in os.listdir(path)
            if f.lower().endswith(IMG_EXTS))
    return [path]


def build_pipeline(cfg):
    scale = cfg.get('img_scale', 640)
    return Compose([
        dict(type='LoadImageFromFile'),
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


def draw(img, boxes, scores, labels, class_names):
    for (x1, y1, x2, y2), s, c in zip(boxes.tolist(), scores.tolist(),
                                      labels.tolist()):
        color = PALETTE[int(c) % len(PALETTE)]
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, color, 2)
        name = class_names[int(c)] if int(c) < len(class_names) else str(c)
        cv2.putText(img, f'{name} {s:.2f}', (p1[0], max(0, p1[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img


@torch.no_grad()
def infer_one(model, pipeline, img_path, score_thr):
    data = pipeline(dict(img_path=img_path, img_id=osp.basename(img_path)))
    batch = pseudo_collate([data])
    sample = model.test_step(batch)[0]
    pred = sample.pred_instances
    scores = pred.scores.cpu().numpy()
    keep = scores >= score_thr
    boxes = pred.bboxes.cpu().numpy()[keep]
    scores = scores[keep]
    labels = pred.labels.cpu().numpy()[keep]
    boxes = restore_boxes(boxes, sample.metainfo)
    return boxes, scores, labels


def main():
    args = parse_args()
    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')

    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmaivision'))
    class_names = list(cfg.get('metainfo', {}).get('classes', ()))

    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.to(device).eval()

    pipeline = build_pipeline(cfg)
    images = collect_images(args.image)
    if not images:
        raise SystemExit(f'未找到图片: {args.image}')
    os.makedirs(args.out_dir, exist_ok=True)

    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            print(f'跳过(读不到): {img_path}')
            continue
        boxes, scores, labels = infer_one(
            model, pipeline, img_path, args.score_thr)
        vis = draw(img, boxes, scores, labels, class_names)
        out_path = osp.join(args.out_dir, osp.basename(img_path))
        cv2.imwrite(out_path, vis)

        from collections import Counter
        cnt = Counter(
            class_names[int(c)] if int(c) < len(class_names) else str(c)
            for c in labels.tolist())
        summary = ', '.join(f'{k}×{v}' for k, v in cnt.most_common()) or '无'
        print(f'{osp.basename(img_path)}: {len(boxes)} 个目标 [{summary}] '
              f'-> {out_path}')


if __name__ == '__main__':
    main()
