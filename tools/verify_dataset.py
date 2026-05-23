"""LabelmeDetDataset 验证脚本：
build 配置中的 dataset，打印关键字段并把 bbox/polygon 渲染到图片保存。

用法：
    python tools/verify_dataset.py <config> [--out-dir vis] [--num 5]
                                             [--split train]
"""
import argparse
import os
import os.path as osp
from collections import Counter

from mmengine.config import Config

import mmaivision  # noqa: F401 触发 registry 注册
from mmaivision.registry import DATASETS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='路径到 dataset config (.py)')
    parser.add_argument(
        '--out-dir', default='vis', help='可视化输出目录')
    parser.add_argument(
        '--num', type=int, default=5, help='可视化前 N 个样本')
    parser.add_argument(
        '--split',
        default='train',
        choices=['train', 'val', 'test'],
        help='使用 cfg 中哪个 dataloader 的 dataset')
    return parser.parse_args()


def render(data_info, classes):
    try:
        import cv2
    except ImportError as e:
        raise SystemExit(
            'opencv-python 未安装，请运行 `pip install opencv-python`'
        ) from e
    img = cv2.imread(data_info['img_path'])
    if img is None:
        print(f"  [warn] 读不到图片: {data_info['img_path']}，跳过渲染")
        return None
    for inst in data_info['instances']:
        x1, y1, x2, y2 = [int(round(v)) for v in inst['bbox']]
        is_diff = inst['ignore_flag'] == 1
        color = (128, 128, 128) if is_diff else (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        if 'mask' in inst:
            import numpy as np
            for poly in inst['mask']:
                pts = np.asarray(poly, dtype=np.int32).reshape(-1, 2)
                cv2.polylines(img, [pts], True, (255, 0, 0), 2)

        label = classes[inst['bbox_label']]
        text = f"{label}{' (diff)' if is_diff else ''}"
        cv2.putText(img, text, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    dataloader_key = f'{args.split}_dataloader'
    dataset_cfg = getattr(cfg, dataloader_key).dataset
    # 把 pipeline 临时清空 —— 我们要拿原始 dict 渲染，不让 transforms 改 bbox
    dataset_cfg = dict(dataset_cfg)
    dataset_cfg['pipeline'] = []
    dataset_cfg['serialize_data'] = False

    ds = DATASETS.build(dataset_cfg)
    classes = ds.metainfo['classes']

    print(f'=== {args.split} dataset ===')
    print(f'  config           : {args.config}')
    print(f'  type             : {dataset_cfg["type"]}')
    print(f'  classes          : {classes}')
    print(f'  len(ds)          : {len(ds)}')

    label_counter = Counter()
    instance_total = 0
    for d in ds.data_list:
        for inst in d['instances']:
            label_counter[classes[inst['bbox_label']]] += 1
            instance_total += 1
    print(f'  total instances  : {instance_total}')
    print(f'  per-class counts : {dict(label_counter)}')

    os.makedirs(args.out_dir, exist_ok=True)
    print(f'\n=== 前 {args.num} 个样本 ===')
    for i in range(min(args.num, len(ds))):
        info = ds.data_list[i]
        n_inst = len(info['instances'])
        preview_bboxes = [inst['bbox'] for inst in info['instances'][:3]]
        print(f'  [{i}] {info["img_id"]} | {info["width"]}x{info["height"]}'
              f' | {n_inst} instances | bboxes[:3]={preview_bboxes}')
        img = render(info, classes)
        if img is not None:
            out_path = osp.join(args.out_dir, f'{info["img_id"]}.jpg')
            import cv2
            cv2.imwrite(out_path, img)
            print(f'        → 已保存 {out_path}')

    print('\n完成。')


if __name__ == '__main__':
    main()
