"""YOLOv5 共享算子,对齐 ultralytics/yolov5 models/common.py。"""
import math
from typing import List, Optional, Union

import torch
import torch.nn as nn


def autopad(k: Union[int, List[int]],
            p: Optional[Union[int, List[int]]] = None,
            d: int = 1) -> Union[int, List[int]]:
    """Same-padding 计算,支持 dilation。"""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


def make_divisible(x: float, divisor: int = 8) -> int:
    """向上取到 divisor 的整数倍,与 ultralytics 一致。"""
    return math.ceil(x / divisor) * divisor


class Conv(nn.Module):
    """Conv + BN + SiLU,YOLOv5 通用卷积块。"""

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1,
                 p: Optional[int] = None, g: int = 1, d: int = 1,
                 act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d),
                              groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))
