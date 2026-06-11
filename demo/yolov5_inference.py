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
from collections import Counter

import cv2
import torch

from common import (build_model, build_pipeline, collect_images,
                    draw_instances, infer_one, load_config)


def parse_args():
    p = argparse.ArgumentParser(description='YOLOv5 配置驱动推理')
    p.add_argument('image', help='输入图片路径或目录')
    p.add_argument('config', help='训练 config 路径')
    p.add_argument('checkpoint', help='模型权重 .pth')
    p.add_argument('--out-dir', default='work_dirs/infer', help='结果输出目录')
    p.add_argument('--score-thr', type=float, default=0.3, help='可视化阈值')
    p.add_argument('--device', default=None, help='cuda:0 / cpu,默认自动')
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')

    cfg, class_names = load_config(args.config)
    model = build_model(cfg, args.checkpoint, device)
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
            model, pipeline, img, osp.basename(img_path), args.score_thr)
        vis = draw_instances(img, boxes, labels, class_names, scores=scores)
        out_path = osp.join(args.out_dir, osp.basename(img_path))
        cv2.imwrite(out_path, vis)

        cnt = Counter(
            class_names[int(c)] if int(c) < len(class_names) else str(c)
            for c in labels.tolist())
        summary = ', '.join(f'{k}×{v}' for k, v in cnt.most_common()) or '无'
        print(f'{osp.basename(img_path)}: {len(boxes)} 个目标 [{summary}] '
              f'-> {out_path}')


if __name__ == '__main__':
    main()
