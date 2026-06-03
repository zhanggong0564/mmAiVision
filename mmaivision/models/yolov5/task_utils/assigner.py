"""YOLOv5 shape-based + 3-grid 扩展 batch assigner。

实现对齐 ultralytics ComputeLoss.build_targets。
"""
from typing import Dict, List, Sequence, Tuple

import torch
from mmengine.structures import InstanceData
from torch import Tensor

from mmaivision.registry import TASK_UTILS


@TASK_UTILS.register_module()
class YOLOv5BatchAssigner:
    """YOLOv5 batch 匹配器。"""

    def __init__(self,
                 num_classes: int,
                 strides: Sequence[int],
                 num_base_priors: int = 3,
                 prior_match_thr: float = 4.0,
                 near_neighbor_thr: float = 0.5):
        if num_classes < 1:
            raise ValueError(f'num_classes 必须 >= 1, got {num_classes}')
        if num_base_priors < 1:
            raise ValueError(
                f'num_base_priors 必须 >= 1, got {num_base_priors}')
        if prior_match_thr <= 0:
            raise ValueError(
                f'prior_match_thr 必须 > 0, got {prior_match_thr}')
        if not (0 < near_neighbor_thr < 1):
            raise ValueError(
                f'near_neighbor_thr 必须 ∈ (0, 1), got {near_neighbor_thr}')
        for s in strides:
            if s <= 0:
                raise ValueError(f'stride 必须 > 0, got {s}')
        self.num_classes = num_classes
        self.strides = list(strides)
        self.num_base_priors = num_base_priors
        self.prior_match_thr = prior_match_thr
        self.near_neighbor_thr = near_neighbor_thr

    @staticmethod
    def _empty_dict(device) -> Dict[str, Tensor]:
        return dict(
            img_idx=torch.zeros(0, dtype=torch.int64, device=device),
            anchor_idx=torch.zeros(0, dtype=torch.int64, device=device),
            grid_y=torch.zeros(0, dtype=torch.int64, device=device),
            grid_x=torch.zeros(0, dtype=torch.int64, device=device),
            gt_xy=torch.zeros(0, 2, device=device),
            gt_wh=torch.zeros(0, 2, device=device),
            gt_class=torch.zeros(0, dtype=torch.int64, device=device),
        )

    def __call__(self,
                 batch_gt_instances: List[InstanceData],
                 anchors: List[Tensor],
                 featmap_sizes: List[Tuple[int, int]]
                 ) -> List[Dict[str, Tensor]]:
        num_levels = len(anchors)
        assert len(self.strides) == num_levels, \
            f'strides 与 anchors 层数必须一致, ' \
            f'got {len(self.strides)} vs {num_levels}'
        device = anchors[0].device

        # 把 batch 所有 gt 拼成 (T, 6):img_idx, class, cx, cy, w, h(像素)
        gt_list = []
        for img_idx, gt in enumerate(batch_gt_instances):
            bboxes = gt.bboxes.to(device)
            labels = gt.labels.to(device)
            if bboxes.numel() == 0:
                continue
            cx = (bboxes[:, 0] + bboxes[:, 2]) / 2
            cy = (bboxes[:, 1] + bboxes[:, 3]) / 2
            w = bboxes[:, 2] - bboxes[:, 0]
            h = bboxes[:, 3] - bboxes[:, 1]
            img_col = torch.full(
                (bboxes.shape[0],), img_idx,
                dtype=torch.float32, device=device)
            gt_list.append(torch.stack(
                [img_col, labels.float(), cx, cy, w, h], dim=-1))
        if not gt_list:
            return [self._empty_dict(device) for _ in range(num_levels)]
        all_gt = torch.cat(gt_list, dim=0)  # (T, 6)
        T = all_gt.shape[0]

        # 加 anchor 索引一维: (na, T, 7)
        na = self.num_base_priors
        anchor_idx_col = torch.arange(
            na, device=device).view(-1, 1, 1).expand(-1, T, 1).float()
        gt_a = torch.cat(
            [all_gt.unsqueeze(0).expand(na, -1, -1), anchor_idx_col], dim=-1)

        off = torch.tensor(
            [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]],
            dtype=torch.float32, device=device) * self.near_neighbor_thr

        results = []
        for layer_idx in range(num_levels):
            anchors_l = anchors[layer_idx].to(device)  # (na, 2) 网格单位 wh
            ny, nx = featmap_sizes[layer_idx]
            stride = self.strides[layer_idx]

            # gt cxcywh 像素 → 网格单位
            gt_layer = gt_a.clone()
            gt_layer[..., 2:6] = gt_layer[..., 2:6] / stride

            # shape match: wh ratio
            wh_ratio = gt_layer[..., 4:6] / anchors_l.view(na, 1, 2)
            max_ratio = torch.maximum(
                wh_ratio, 1.0 / wh_ratio).max(dim=-1).values
            keep = max_ratio < self.prior_match_thr  # (na, T)
            gt_kept = gt_layer[keep]  # (M0, 7)

            if gt_kept.shape[0] == 0:
                results.append(self._empty_dict(device))
                continue

            # 3-grid 扩展:对每个 kept gt,选 ≤2 个邻近 grid
            gxy = gt_kept[:, 2:4]  # cxcy 网格单位
            gxy_inv = torch.tensor(
                [nx, ny], device=device, dtype=torch.float32) - gxy
            j_pos = ((gxy % 1 < self.near_neighbor_thr) & (gxy > 1)).T
            k_pos = ((gxy_inv % 1 < self.near_neighbor_thr) & (gxy_inv > 1)).T
            mask = torch.stack(
                [torch.ones_like(j_pos[0]), j_pos[0], j_pos[1],
                 k_pos[0], k_pos[1]], dim=0)  # (5, M0)
            gt_ext = gt_kept.repeat(5, 1, 1)[mask]  # (M, 7)
            off_ext = off[:, None, :].repeat(1, gt_kept.shape[0], 1)[mask]

            gxy_ext = gt_ext[:, 2:4]
            grid_xy_int = (gxy_ext - off_ext).long()
            grid_x = grid_xy_int[:, 0].clamp(0, nx - 1)
            grid_y = grid_xy_int[:, 1].clamp(0, ny - 1)

            results.append(dict(
                img_idx=gt_ext[:, 0].long(),
                anchor_idx=gt_ext[:, 6].long(),
                grid_y=grid_y,
                grid_x=grid_x,
                gt_xy=gxy_ext,
                gt_wh=gt_ext[:, 4:6],
                gt_class=gt_ext[:, 1].long(),
            ))
        return results
