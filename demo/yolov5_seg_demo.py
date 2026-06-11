"""YOLOv5-seg 实例分割推理脚本(配置驱动,走模型内建 predict)。

与 yolov5_inference.py 一致的脚手架,额外把每个实例 mask 还原回原图分辨率
并半透明上色叠加。

用法:
    python demo/yolov5_seg_demo.py <img|dir> configs/yolov5_n_seg_labelme.py \
        work_dirs/yolov5_n_seg_labelme/best_labelme_seg_mAP_50_epoch_100.pth \
        --out-dir work_dirs/infer_seg --score-thr 0.3
"""
import argparse
import os
import os.path as osp

import cv2
import torch

from common import (build_model, build_pipeline, collect_images,
                    draw_instances, infer_one, load_config)


def parse_args():
    p = argparse.ArgumentParser(description='YOLOv5-seg 配置驱动推理')
    p.add_argument('image', help='输入图片路径或目录')
    p.add_argument('config', help='训练 config 路径')
    p.add_argument('checkpoint', help='模型权重 .pth')
    p.add_argument('--out-dir', default='work_dirs/infer_seg',
                   help='结果输出目录')
    p.add_argument('--score-thr', type=float, default=0.3, help='可视化阈值')
    p.add_argument('--alpha', type=float, default=0.5, help='mask 叠加透明度')
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
        boxes, scores, labels, masks = infer_one(
            model, pipeline, img, osp.basename(img_path), args.score_thr,
            with_masks=True)
        vis = draw_instances(img, boxes, labels, class_names,
                             scores=scores, masks=masks, alpha=args.alpha)
        out_path = osp.join(args.out_dir, osp.basename(img_path))
        cv2.imwrite(out_path, vis)
        print(f'{osp.basename(img_path)}: {len(boxes)} 个实例 -> {out_path}')


if __name__ == '__main__':
    main()
