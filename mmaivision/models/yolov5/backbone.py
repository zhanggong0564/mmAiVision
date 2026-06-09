"""YOLOv5 CSPDarknet backbone。"""
from typing import Sequence, Tuple

from mmengine.model import BaseModule
from torch import Tensor, nn
from torch.nn.modules.batchnorm import _BatchNorm

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
        frozen_stages: 冻结到第几阶段(含)。-1 不冻结;0 冻 stem;
            1 冻 stem+stage1;...;4 冻整个 backbone。被冻结层 requires_grad=False
            且 BN 切 eval(不更新滑动均值)。
        norm_eval: 为 True 时训练阶段把所有 BN 置 eval(冻结 BN 统计),
            小数据集微调常用以避免 BN 统计被带偏。
    """

    BASE_CHANNELS = (64, 128, 256, 512, 1024)
    BASE_N = (3, 6, 9, 3)

    def __init__(self,
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 out_indices: Sequence[int] = (2, 3, 4),
                 frozen_stages: int = -1,
                 norm_eval: bool = False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        if deepen_factor <= 0 or widen_factor <= 0:
            raise ValueError(
                f'deepen_factor / widen_factor 必须 > 0, got '
                f'{deepen_factor=}, {widen_factor=}')
        if not set(out_indices).issubset({2, 3, 4}):
            raise ValueError(
                f'out_indices 必须是 (2,3,4) 子集, got {out_indices}')
        if not -1 <= frozen_stages <= 4:
            raise ValueError(
                f'frozen_stages 必须在 [-1, 4], got {frozen_stages}')
        self.out_indices = tuple(out_indices)
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval

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

        self._freeze_stages()

    def _freeze_stages(self) -> None:
        """冻结 stem 及前 frozen_stages 个 stage:停梯度 + BN 转 eval。"""
        if self.frozen_stages < 0:
            return
        # 索引 0=stem, 1..4=stage1..stage4,冻结 [0, frozen_stages]
        stages = [self.stem, self.stage1, self.stage2, self.stage3,
                  self.stage4]
        for m in stages[:self.frozen_stages + 1]:
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True) -> 'YOLOv5CSPDarknet':
        """重写以保证冻结层始终 eval,并按需冻结所有 BN 统计。"""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
        return self

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
