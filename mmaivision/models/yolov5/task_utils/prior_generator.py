"""YOLOv5 anchor 生成器:固定 anchors + 按 featmap_size 生成 grid。"""
from typing import List, Sequence, Tuple

import torch
from torch import Tensor

from mmaivision.registry import TASK_UTILS


@TASK_UTILS.register_module()
class YOLOv5AnchorGenerator:
    """YOLOv5 anchor 生成器。

    Args:
        base_sizes: 每层的 anchor 列表,如 ``[[(w0,h0),(w1,h1),(w2,h2)], ...]``,
            像素单位。
        strides: 每层 stride,如 ``[8, 16, 32]``。
    """

    def __init__(self,
                 base_sizes: Sequence[Sequence[Tuple[int, int]]],
                 strides: Sequence[int]):
        assert len(base_sizes) == len(strides), \
            f'base_sizes 和 strides 长度必须一致, got ' \
            f'{len(base_sizes)} vs {len(strides)}'
        n_per_level = [len(b) for b in base_sizes]
        assert len(set(n_per_level)) == 1, \
            f'每层 anchor 数必须一致, got {n_per_level}'
        for s in strides:
            if s <= 0:
                raise ValueError(f'stride 必须 > 0, got {s}')
        self.base_sizes = [
            torch.tensor(b, dtype=torch.float32) for b in base_sizes]
        self.strides = list(strides)
        self._num_base_priors = n_per_level[0]

    @property
    def num_levels(self) -> int:
        return len(self.strides)

    @property
    def num_base_priors(self) -> List[int]:
        return [self._num_base_priors] * self.num_levels

    def grid_priors(self,
                    featmap_sizes: Sequence[Tuple[int, int]],
                    device='cpu',
                    dtype=torch.float32) -> List[Tensor]:
        """每层返回 ``(na, ny, nx, 2)`` 的 anchor wh,单位:网格(stride 单位)。"""
        assert len(featmap_sizes) == self.num_levels
        outs = []
        for i, (ny, nx) in enumerate(featmap_sizes):
            anchors_grid = self.base_sizes[i].to(device=device, dtype=dtype) \
                / self.strides[i]
            # (na, 2) -> (na, ny, nx, 2)
            anchors_grid = anchors_grid.view(-1, 1, 1, 2).expand(
                -1, ny, nx, -1).contiguous()
            outs.append(anchors_grid)
        return outs

    def grid_xy(self,
                featmap_sizes: Sequence[Tuple[int, int]],
                device='cpu',
                dtype=torch.float32) -> List[Tensor]:
        """每层返回 ``(ny, nx, 2)`` 的 grid 中心坐标(网格单位)。"""
        assert len(featmap_sizes) == self.num_levels
        outs = []
        for ny, nx in featmap_sizes:
            ys, xs = torch.meshgrid(
                torch.arange(ny, device=device, dtype=dtype),
                torch.arange(nx, device=device, dtype=dtype),
                indexing='ij')
            grid = torch.stack([xs, ys], dim=-1)  # (ny, nx, 2) [x, y]
            outs.append(grid)
        return outs
