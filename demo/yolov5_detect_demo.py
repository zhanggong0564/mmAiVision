"""YOLOv5 推理 demo：用转换后的官方权重对单张图做完整检测。

本仓库 YOLOv5Head 只输出原始 conv 特征，这里补上标准 YOLOv5 解码
（sigmoid + grid/anchor 还原 + NMS），用于验证转换权重能真正检测目标。

用法：
    python demo/yolov5_detect_demo.py <image> --weights work_dirs/yolov5n_official.pth \
        --size n --out-file work_dirs/det_result.jpg
"""
import argparse

import cv2
import numpy as np
import torch
from torchvision.ops import batched_nms

import mmaivision  # noqa: F401  触发 registry 注册
from mmaivision.models.yolov5.common import make_divisible
from mmaivision.models.yolov5.task_utils import YOLOv5AnchorGenerator
from mmaivision.registry import MODELS

SIZE_FACTORS = {'n': (0.33, 0.25), 's': (0.33, 0.50), 'm': (0.67, 0.75),
                'l': (1.00, 1.00), 'x': (1.33, 1.25)}
BASE_CHANNELS = (64, 128, 256, 512, 1024)

# YOLOv5 默认 anchors（像素）与 strides
ANCHORS = [[(10, 13), (16, 30), (33, 23)],
           [(30, 61), (62, 45), (59, 119)],
           [(116, 90), (156, 198), (373, 326)]]
STRIDES = [8, 16, 32]

COCO = (
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('image', help='输入图片路径')
    p.add_argument('--weights', required=True, help='转换后的 .pth 权重')
    p.add_argument('--size', default='n', choices=list(SIZE_FACTORS))
    p.add_argument('--num-classes', type=int, default=80)
    p.add_argument('--img-size', type=int, default=640)
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--iou', type=float, default=0.45)
    p.add_argument('--out-file', default='work_dirs/det_result.jpg')
    return p.parse_args()


def build_model(size, num_classes, weights):
    d, w = SIZE_FACTORS[size]
    ch = [make_divisible(c * w, 8) for c in BASE_CHANNELS]
    p = [ch[2], ch[3], ch[4]]
    model = MODELS.build(dict(
        type='YOLOv5Detector',
        backbone=dict(type='YOLOv5CSPDarknet', deepen_factor=d, widen_factor=w),
        neck=dict(type='YOLOv5PAFPN', in_channels=p, out_channels=p,
                  deepen_factor=d, widen_factor=w),
        head=dict(type='YOLOv5Head', num_classes=num_classes, in_channels=p)))
    ckpt = torch.load(weights, map_location='cpu')
    model.load_state_dict(ckpt.get('state_dict', ckpt))
    return model.cpu().eval()


def letterbox(img, new=640, color=114):
    h, w = img.shape[:2]
    r = min(new / h, new / w)
    nh, nw = round(h * r), round(w * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((new, new, 3), color, dtype=np.uint8)
    top, left = (new - nh) // 2, (new - nw) // 2
    out[top:top + nh, left:left + nw] = resized
    return out, r, left, top


@torch.no_grad()
def detect(model, img, num_classes, img_size, conf_thr, iou_thr):
    lb, r, left, top = letterbox(img, img_size)
    rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    inp = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)

    feats = model(inp, mode='tensor')                 # list (1, na*no, h, w)
    fsizes = [(f.shape[2], f.shape[3]) for f in feats]
    gen = YOLOv5AnchorGenerator(base_sizes=ANCHORS, strides=STRIDES)
    anchor_grid = gen.grid_priors(fsizes)             # (na, ny, nx, 2) 网格单位
    grid_xy = gen.grid_xy(fsizes)                     # (ny, nx, 2)

    na, no = 3, num_classes + 5
    all_box, all_score, all_label = [], [], []
    for i, f in enumerate(feats):
        b, _, ny, nx = f.shape
        x = f.view(b, na, no, ny, nx).permute(0, 1, 3, 4, 2).sigmoid()
        g = grid_xy[i].view(1, 1, ny, nx, 2)
        a = anchor_grid[i].view(1, na, ny, nx, 2)
        xy = (x[..., 0:2] * 2 - 0.5 + g) * STRIDES[i]
        wh = (x[..., 2:4] * 2) ** 2 * a * STRIDES[i]
        obj = x[..., 4:5]
        cls_conf, cls_id = x[..., 5:].max(-1, keepdim=True)
        score = (obj * cls_conf).view(-1)
        box = torch.cat([xy, wh], -1).view(-1, 4)
        all_box.append(box)
        all_score.append(score)
        all_label.append(cls_id.view(-1))

    box = torch.cat(all_box)
    score = torch.cat(all_score)
    label = torch.cat(all_label)
    keep = score > conf_thr
    box, score, label = box[keep], score[keep], label[keep]

    # xywh -> xyxy
    xyxy = torch.empty_like(box)
    xyxy[:, 0] = box[:, 0] - box[:, 2] / 2
    xyxy[:, 1] = box[:, 1] - box[:, 3] / 2
    xyxy[:, 2] = box[:, 0] + box[:, 2] / 2
    xyxy[:, 3] = box[:, 1] + box[:, 3] / 2

    keep = batched_nms(xyxy, score, label, iou_thr)
    xyxy, score, label = xyxy[keep], score[keep], label[keep]

    # 还原到原图坐标（去 letterbox 偏移与缩放）
    xyxy[:, [0, 2]] -= left
    xyxy[:, [1, 3]] -= top
    xyxy /= r
    h, w = img.shape[:2]
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clamp(0, w)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clamp(0, h)
    return xyxy, score, label


def draw(img, boxes, scores, labels):
    for (x1, y1, x2, y2), s, c in zip(boxes.tolist(), scores.tolist(),
                                      labels.tolist()):
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, (0, 255, 0), 2)
        text = f'{COCO[int(c)]} {s:.2f}'
        cv2.putText(img, text, (p1[0], max(0, p1[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return img


def main():
    args = parse_args()
    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f'读不到图片: {args.image}')
    model = build_model(args.size, args.num_classes, args.weights)
    boxes, scores, labels = detect(
        model, img, args.num_classes, args.img_size, args.conf, args.iou)

    print(f'检测到 {len(boxes)} 个目标:')
    from collections import Counter
    cnt = Counter(COCO[int(c)] for c in labels.tolist())
    for name, n in cnt.most_common():
        print(f'  {name}: {n}')
    for (x1, y1, x2, y2), s, c in zip(boxes.tolist(), scores.tolist(),
                                      labels.tolist()):
        print(f'  [{COCO[int(c)]:12s}] conf={s:.3f} '
              f'box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})')

    import os
    os.makedirs(os.path.dirname(args.out_file) or '.', exist_ok=True)
    cv2.imwrite(args.out_file, draw(img, boxes, scores, labels))
    print(f'已保存可视化 -> {args.out_file}')


if __name__ == '__main__':
    main()
