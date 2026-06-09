"""数据增强可视化:把训练 pipeline 增强后的图像连同 bbox / mask 叠加输出。

用于上线前肉眼检查 Mosaic / 仿射 / 翻转等增强是否破坏标注一致性(尤其线标这类
细长目标在翻转 / 旋转后的朝向)。直接从配置构建训练集并按下标取样,因此会完整跑一
遍训练 pipeline(含 Mosaic / MixUp 的多图取样,dataset 引用已由 prepare_data 注入)。

用法:
    # 用配置里现有的 train pipeline 可视化
    python demo/visualize_augmentation.py configs/yolov5_n_seg_labelme.py \
        --out-dir work_dirs/aug_vis --num 20

    # 不改配置,直接预览“完整增强 pipeline”(Mosaic+Affine+MixUp+HSV+Flip)
    python demo/visualize_augmentation.py configs/yolov5_n_seg_labelme.py \
        --out-dir work_dirs/aug_vis --num 20 --aug
"""
import argparse
import os
import os.path as osp

import cv2
import numpy as np
from mmengine.config import Config
from mmengine.registry import init_default_scope

import mmaivision  # noqa: F401  触发 registry 注册
from mmaivision.registry import DATASETS

PALETTE = [(0, 255, 0), (0, 128, 255), (255, 0, 0), (0, 0, 255),
           (255, 0, 255), (255, 255, 0), (128, 0, 255), (0, 215, 255)]


def parse_args():
    p = argparse.ArgumentParser(description='数据增强可视化')
    p.add_argument('config', help='训练 config 路径')
    p.add_argument('--out-dir', default='work_dirs/aug_vis',
                   help='结果输出目录')
    p.add_argument('--num', type=int, default=20, help='可视化样本数')
    p.add_argument('--start', type=int, default=0, help='起始样本下标')
    p.add_argument('--alpha', type=float, default=0.5, help='mask 叠加透明度')
    p.add_argument('--aug', action='store_true',
                   help='用内置完整增强 pipeline 覆盖配置(便于预览未写入配置的增强)')
    return p.parse_args()


def build_full_aug_pipeline(img_scale):
    """内置“完整增强” pipeline,用于 --aug 预览(与建议配置一致)。"""
    pre_transform = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadLabelmeAnnotations'),
    ]
    return [
        dict(type='LoadImageFromFile'),
        dict(type='LoadLabelmeAnnotations'),
        dict(type='Mosaic', img_scale=img_scale, pre_transform=pre_transform),
        dict(type='RandomAffine',
             border=(-img_scale // 2, -img_scale // 2),
             max_translate_ratio=0.1, scaling_ratio_range=(0.5, 1.5)),
        dict(type='MixUp', pre_transform=pre_transform, prob=0.5),
        dict(type='YOLOv5HSVRandomAug'),
        dict(type='RandomFlip', prob=0.5),
        dict(type='PackDetInputs'),
    ]


def to_bgr_image(inputs):
    """PackDetInputs 的 inputs(CHW BGR uint8 张量)→ HWC uint8 ndarray。"""
    img = inputs.numpy() if hasattr(inputs, 'numpy') else np.asarray(inputs)
    return np.ascontiguousarray(img.transpose(1, 2, 0))


def draw(img, bboxes, labels, masks, class_names, alpha):
    overlay = img.copy()
    for box, c, m in zip(bboxes, labels, masks):
        color = PALETTE[int(c) % len(PALETTE)]
        if m is not None and m.any():
            overlay[m] = color
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        name = class_names[int(c)] if int(c) < len(class_names) else str(c)
        cv2.putText(img, name, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    return img


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmaivision'))
    class_names = list(cfg.get('metainfo', {}).get('classes', ()))
    img_scale = cfg.get('img_scale', 640)

    ds_cfg = cfg.train_dataloader.dataset
    if args.aug:
        ds_cfg = dict(ds_cfg)
        ds_cfg['pipeline'] = build_full_aug_pipeline(img_scale)
        ds_cfg.pop('indices', None)  # 预览不限子集
    dataset = DATASETS.build(ds_cfg)

    os.makedirs(args.out_dir, exist_ok=True)
    n = min(args.num, len(dataset))
    print(f'数据集共 {len(dataset)} 张,可视化 {n} 张 -> {args.out_dir}'
          f'{"(完整增强预览)" if args.aug else "(配置 pipeline)"}')

    for k in range(n):
        idx = (args.start + k) % len(dataset)
        data = dataset[idx]
        img = to_bgr_image(data['inputs'])
        gi = data['data_samples'].gt_instances
        bboxes = gi.bboxes.numpy() if hasattr(gi.bboxes, 'numpy') else \
            np.asarray(gi.bboxes)
        labels = gi.labels.numpy() if hasattr(gi.labels, 'numpy') else \
            np.asarray(gi.labels)
        if 'masks' in gi:
            masks = gi.masks.numpy().astype(bool)
        else:
            masks = [None] * len(bboxes)
        vis = draw(img.copy(), bboxes, labels, masks, class_names, args.alpha)
        out_path = osp.join(args.out_dir, f'aug_{idx:05d}.jpg')
        cv2.imwrite(out_path, vis)
        print(f'  [{idx}] {len(bboxes)} 个实例 -> {out_path}')


if __name__ == '__main__':
    main()
