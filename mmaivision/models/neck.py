"""YOLOv5 PAN-FPN neck。"""
from typing import Sequence, Tuple

import torch
from mmengine.model import BaseModule
from torch import Tensor, nn

from mmaivision.registry import MODELS
from .common import C3, Conv


@MODELS.register_module()
class YOLOv5PAFPN(BaseModule):
    """YOLOv5 PAN-FPN,top-down + bottom-up,输出与输入同层数。

    Args:
        in_channels: 三个输入通道。
        out_channels: 三个输出通道,通常与 in_channels 一致。
        deepen_factor: C3 块数乘子,与 backbone 一致。
        widen_factor: 通道乘子,仅用于与 backbone 同步上下文,本模块不内部使用。
        num_csp_blocks: C3 中 Bottleneck 的基础数量,默认 3(ultralytics 标配)。
    """

    def __init__(self,
                 in_channels: Sequence[int],
                 out_channels: Sequence[int],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 num_csp_blocks: int = 3,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        assert len(in_channels) == 3, \
            f'in_channels 长度必须为 3, got {len(in_channels)}'
        assert len(out_channels) == 3, \
            f'out_channels 长度必须为 3, got {len(out_channels)}'
        self.in_channels = list(in_channels)
        self.out_channels = list(out_channels)
        n = max(round(num_csp_blocks * deepen_factor), 1)

        # Top-down:P5 → 1x1 reduce → up → cat P4 → C3 → M4
        self.reduce_p5 = Conv(in_channels[2], in_channels[1], k=1, s=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.top_down_c3_p4 = C3(
            in_channels[1] * 2, in_channels[1], n=n, shortcut=False)

        # M4 → 1x1 reduce → up → cat P3 → C3 → N3
        self.reduce_p4 = Conv(in_channels[1], in_channels[0], k=1, s=1)
        self.top_down_c3_p3 = C3(
            in_channels[0] * 2, out_channels[0], n=n, shortcut=False)

        # Bottom-up:N3 → 3x3 s=2 → cat (reduce_p4 输出) → C3 → N4
        self.downsample_n3 = Conv(out_channels[0], out_channels[0], k=3, s=2)
        self.bottom_up_c3_n4 = C3(
            out_channels[0] + in_channels[0], out_channels[1],
            n=n, shortcut=False)

        # N4 → 3x3 s=2 → cat (reduce_p5 输出) → C3 → N5
        self.downsample_n4 = Conv(out_channels[1], out_channels[1], k=3, s=2)
        self.bottom_up_c3_n5 = C3(
            out_channels[1] + in_channels[1], out_channels[2],
            n=n, shortcut=False)

    def forward(self, feats: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        p3, p4, p5 = feats

        # Top-down
        p5_reduced = self.reduce_p5(p5)
        m4 = self.top_down_c3_p4(
            torch.cat([self.upsample(p5_reduced), p4], dim=1))
        m4_reduced = self.reduce_p4(m4)
        n3 = self.top_down_c3_p3(
            torch.cat([self.upsample(m4_reduced), p3], dim=1))

        # Bottom-up
        n4 = self.bottom_up_c3_n4(
            torch.cat([self.downsample_n3(n3), m4_reduced], dim=1))
        n5 = self.bottom_up_c3_n5(
            torch.cat([self.downsample_n4(n4), p5_reduced], dim=1))

        return n3, n4, n5
