"""YOLOv5 CSPDarknet backbone。"""
from typing import Sequence, Tuple

from mmengine.model import BaseModule
from torch import Tensor, nn

from mmaivision.registry import MODELS
from .common import C3, SPPF, Conv, make_divisible


@MODELS.register_module()
class YOLOv5CSPDarknet(BaseModule):
    """YOLOv5 CSPDarknet,输出 P3/P4/P5 三层特征。

    基础通道 [64,128,256,512,1024],C3 块数配方 [3,6,9,3],SPPF 放在最后。

    Args:
        deepen_factor: C3 块数 n 的乘子(对应 ultralytics depth_multiple)。
        widen_factor: 通道数 c 的乘子(对应 ultralytics width_multiple)。
        out_indices: 取哪几个 stage 作为输出,默认 (2, 3, 4) → P3/P4/P5。
    """

    BASE_CHANNELS = (64, 128, 256, 512, 1024)
    BASE_N = (3, 6, 9, 3)

    def __init__(self,
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 out_indices: Sequence[int] = (2, 3, 4),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        if deepen_factor <= 0 or widen_factor <= 0:
            raise ValueError(
                f'deepen_factor / widen_factor 必须 > 0, got '
                f'{deepen_factor=}, {widen_factor=}')
        if not set(out_indices).issubset({2, 3, 4}):
            raise ValueError(
                f'out_indices 必须是 (2,3,4) 子集, got {out_indices}')
        self.out_indices = tuple(out_indices)

        channels = [make_divisible(c * widen_factor, 8)
                    for c in self.BASE_CHANNELS]
        n_blocks = [max(round(n * deepen_factor), 1) for n in self.BASE_N]

        # stem: Conv 6x6 s=2,p=2 等价 ultralytics 写法
        self.stem = Conv(3, channels[0], k=6, s=2, p=2)

        # stage 1: 输出 stride 4
        self.stage1 = nn.Sequential(
            Conv(channels[0], channels[1], k=3, s=2),
            C3(channels[1], channels[1], n=n_blocks[0]))
        # stage 2: 输出 stride 8 (P3)
        self.stage2 = nn.Sequential(
            Conv(channels[1], channels[2], k=3, s=2),
            C3(channels[2], channels[2], n=n_blocks[1]))
        # stage 3: 输出 stride 16 (P4)
        self.stage3 = nn.Sequential(
            Conv(channels[2], channels[3], k=3, s=2),
            C3(channels[3], channels[3], n=n_blocks[2]))
        # stage 4: 输出 stride 32 (P5),末尾加 SPPF
        self.stage4 = nn.Sequential(
            Conv(channels[3], channels[4], k=3, s=2),
            C3(channels[4], channels[4], n=n_blocks[3]),
            SPPF(channels[4], channels[4], k=5))

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        outs = []
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        if 2 in self.out_indices:
            outs.append(x)
        x = self.stage3(x)
        if 3 in self.out_indices:
            outs.append(x)
        x = self.stage4(x)
        if 4 in self.out_indices:
            outs.append(x)
        return tuple(outs)
