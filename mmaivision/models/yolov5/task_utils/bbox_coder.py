"""YOLOv5 bbox 编解码:中心点 sigmoid*2-0.5,宽高 (sigmoid*2)^2 * anchor。"""
import torch
from torch import Tensor

from mmaivision.registry import TASK_UTILS


@TASK_UTILS.register_module()
class YOLOv5BBoxCoder:
    """YOLOv5 bbox 编解码。

    decode: pred (logits) → xyxy 像素;encode: gt_xywh 像素 → tx/ty/tw/th。
    """

    def decode(self,
               pred: Tensor,
               anchor: Tensor,
               grid_xy: Tensor,
               stride: int) -> Tensor:
        """解码到 xyxy 像素。

        Args:
            pred: ``(B, na, ny, nx, 4)`` 的 raw logits。
            anchor: ``(na, 2)`` 网格单位 wh。
            grid_xy: ``(ny, nx, 2)`` 网格中心 [x, y]。
            stride: int,该层 stride。
        """
        sig = torch.sigmoid(pred)
        # xy: (sig*2 - 0.5 + grid_xy) * stride
        xy = (sig[..., 0:2] * 2 - 0.5 + grid_xy.unsqueeze(0).unsqueeze(0)) \
            * stride
        # wh: (sig*2)^2 * anchor * stride
        wh = (sig[..., 2:4] * 2) ** 2 \
            * anchor.view(1, -1, 1, 1, 2) * stride
        x1 = xy[..., 0:1] - wh[..., 0:1] / 2
        y1 = xy[..., 1:2] - wh[..., 1:2] / 2
        x2 = xy[..., 0:1] + wh[..., 0:1] / 2
        y2 = xy[..., 1:2] + wh[..., 1:2] / 2
        return torch.cat([x1, y1, x2, y2], dim=-1)

    def encode(self,
               gt_xywh: Tensor,
               matched_anchor: Tensor,
               matched_grid_xy: Tensor,
               stride: int) -> Tensor:
        """编码 gt_xywh 像素 → tx/ty/tw/th(decode 的逆运算,返回 logits 前值)。

        Args:
            gt_xywh: ``(M, 4)`` cxcywh 像素。
            matched_anchor: ``(M, 2)`` 网格单位 wh。
            matched_grid_xy: ``(M, 2)`` 网格中心 [x, y]。
            stride: int。
        """
        cx_grid = gt_xywh[:, 0] / stride - matched_grid_xy[:, 0]
        cy_grid = gt_xywh[:, 1] / stride - matched_grid_xy[:, 1]
        sig_x = (cx_grid + 0.5) / 2
        sig_y = (cy_grid + 0.5) / 2
        tx = torch.logit(sig_x.clamp(1e-6, 1 - 1e-6))
        ty = torch.logit(sig_y.clamp(1e-6, 1 - 1e-6))
        w_grid = gt_xywh[:, 2] / stride
        h_grid = gt_xywh[:, 3] / stride
        sig_w = torch.sqrt(w_grid / matched_anchor[:, 0]) / 2
        sig_h = torch.sqrt(h_grid / matched_anchor[:, 1]) / 2
        tw = torch.logit(sig_w.clamp(1e-6, 1 - 1e-6))
        th = torch.logit(sig_h.clamp(1e-6, 1 - 1e-6))
        return torch.stack([tx, ty, tw, th], dim=-1)
