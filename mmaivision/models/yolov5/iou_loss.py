"""YOLOv5 用 CIoU 实现(对齐 ultralytics)。"""
import math

import torch
from torch import Tensor


def bbox_ciou(pred_xyxy: Tensor,
              target_xyxy: Tensor,
              eps: float = 1e-7) -> Tensor:
    """逐对 CIoU。

    Args:
        pred_xyxy: ``(N, 4)`` xyxy。
        target_xyxy: ``(N, 4)`` xyxy。
        eps: 数值稳定项。

    Returns:
        ``(N,)`` CIoU,范围 [-1, 1]。
    """
    px1, py1, px2, py2 = pred_xyxy.unbind(-1)
    tx1, ty1, tx2, ty2 = target_xyxy.unbind(-1)
    inter_x1 = torch.maximum(px1, tx1)
    inter_y1 = torch.maximum(py1, ty1)
    inter_x2 = torch.minimum(px2, tx2)
    inter_y2 = torch.minimum(py2, ty2)
    inter = (inter_x2 - inter_x1).clamp(min=0) \
        * (inter_y2 - inter_y1).clamp(min=0)
    p_area = (px2 - px1) * (py2 - py1)
    t_area = (tx2 - tx1) * (ty2 - ty1)
    union = p_area + t_area - inter + eps
    iou = inter / union

    p_cx = (px1 + px2) / 2
    p_cy = (py1 + py2) / 2
    t_cx = (tx1 + tx2) / 2
    t_cy = (ty1 + ty2) / 2
    center_dist_sq = (p_cx - t_cx) ** 2 + (p_cy - t_cy) ** 2

    enc_x1 = torch.minimum(px1, tx1)
    enc_y1 = torch.minimum(py1, ty1)
    enc_x2 = torch.maximum(px2, tx2)
    enc_y2 = torch.maximum(py2, ty2)
    enc_diag_sq = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + eps

    pw = (px2 - px1).clamp(min=eps)
    ph = (py2 - py1).clamp(min=eps)
    tw = (tx2 - tx1).clamp(min=eps)
    th = (ty2 - ty1).clamp(min=eps)
    v = (4 / math.pi ** 2) * (torch.atan(tw / th) - torch.atan(pw / ph)) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    return iou - center_dist_sq / enc_diag_sq - alpha * v
