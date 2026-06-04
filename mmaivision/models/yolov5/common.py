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
        if isinstance(k, int):
            k = d * (k - 1) + 1
        else:
            k = [d * (x - 1) + 1 for x in k]
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
        # eps/momentum 对齐 ultralytics initialize_weights(BN eps=1e-3,
        # momentum=0.03),保证官方权重转换后数值完全一致。
        self.bn = nn.BatchNorm2d(c2, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """标准 bottleneck: 1x1 + 3x3 + 可选 shortcut。"""

    def __init__(self, c1: int, c2: int, shortcut: bool = True,
                 g: int = 1, e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k=1, s=1)
        self.cv2 = Conv(c_, c2, k=3, s=1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions。"""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True,
                 g: int = 1, e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k=1, s=1)
        self.cv2 = Conv(c1, c_, k=1, s=1)
        self.cv3 = Conv(2 * c_, c2, k=1)
        self.m = nn.Sequential(*(
            Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPPF(nn.Module):
    """快速 SPP,3 次串行 maxpool 替代 SPP 的并行 5/9/13。"""

    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, k=1, s=1)
        self.cv2 = Conv(c_ * 4, c2, k=1, s=1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), dim=1))


class Proto(nn.Module):
    """YOLOv5-seg 原型 mask 网络,对齐 ultralytics models/common.py Proto。

    输入某层特征(通常 P3),输出 ``c2`` 张原型 mask,空间分辨率为输入的 2 倍。
    """

    def __init__(self, c1: int, c_: int = 256, c2: int = 32):
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)  # 默认 1x1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))
