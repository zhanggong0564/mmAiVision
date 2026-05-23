"""YOLOv5 Detect head 的纯网络部分(无 decode / nms)。"""
from typing import List, Sequence, Tuple

import torch.nn as nn
from mmengine.model import BaseModule
from torch import Tensor

from mmaivision.registry import MODELS


@MODELS.register_module()
class YOLOv5Head(BaseModule):
    """YOLOv5 Detect head:三层独立 1x1 Conv,输出原始 feature。

    本模块只交付结构,不做 anchor decode / sigmoid / nms,
    这些留给后续的 postprocess 模块。

    Args:
        num_classes: 类别数 nc。
        in_channels: 三个输入通道,通常等于 neck.out_channels。
        num_base_priors: 每层 anchor 数,默认 3(YOLOv5 标配)。
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: Sequence[int],
                 num_base_priors: int = 3,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        assert len(in_channels) == 3, \
            f'in_channels 长度必须为 3, got {len(in_channels)}'
        if num_classes < 1:
            raise ValueError(f'num_classes 必须 >= 1, got {num_classes}')
        if num_base_priors < 1:
            raise ValueError(
                f'num_base_priors 必须 >= 1, got {num_base_priors}')
        self.num_classes = num_classes
        self.num_base_priors = num_base_priors
        out_c = num_base_priors * (num_classes + 5)
        self.convs = nn.ModuleList(
            [nn.Conv2d(c, out_c, kernel_size=1) for c in in_channels])

    def forward(self, feats: Tuple[Tensor, ...]) -> List[Tensor]:
        return [conv(f) for conv, f in zip(self.convs, feats)]
