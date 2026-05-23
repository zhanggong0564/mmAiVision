# YOLOv5 Network Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现对齐 ultralytics 风格的 YOLOv5 backbone / neck / head 三件套 + Detector wrapper,支持 n/s/m/l/x 全家族变体,只交付网络结构本身(无 loss/postprocess)。

**Architecture:** 共享算子(Conv/C3/SPPF/Bottleneck/make_divisible)放 `common.py`,backbone/neck/head/detector 各一个文件,均通过 `@MODELS.register_module()` 注册到 `mmaivision.registry.MODELS`。三者继承 `BaseModule`,detector 继承 `BaseModel` 并只实现 `mode='tensor'` 分支。通过 `deepen_factor / widen_factor` 参数化全家族,不引入 ultralytics yaml 解析。

**Tech Stack:** PyTorch, mmengine (BaseModule / BaseModel / Registry), pytest

---

## 文件结构

| 路径 | 职责 |
|------|------|
| `mmaivision/models/common.py` | 共享算子:autopad / make_divisible / Conv / Bottleneck / C3 / SPPF |
| `mmaivision/models/backbone.py` | `YOLOv5CSPDarknet` |
| `mmaivision/models/neck.py` | `YOLOv5PAFPN` |
| `mmaivision/models/head.py` | `YOLOv5Head` |
| `mmaivision/models/detector.py` | `YOLOv5Detector` (BaseModel wrapper) |
| `mmaivision/models/__init__.py` | 触发注册 |
| `tests/test_models.py` | 全部 11 个测试用例 |

参考设计:[docs/superpowers/specs/2026-05-23-yolov5-network-design.md](../specs/2026-05-23-yolov5-network-design.md)

---

## Task 1: 创建 common.py 和共享算子(autopad / make_divisible / Conv)

**Files:**
- Create: `mmaivision/models/common.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: 写 test_conv_shape 测试**

```python
# tests/test_models.py
"""Tests for YOLOv5 network components."""
import pytest
import torch
from torch import nn


class TestConv:
    def test_conv_shape(self):
        from mmaivision.models.common import Conv
        layer = Conv(3, 16, k=3, s=2)
        out = layer(torch.randn(1, 3, 64, 64))
        assert out.shape == (1, 16, 32, 32)
        # BN 存在
        assert isinstance(layer.bn, nn.BatchNorm2d)
        # 激活是 SiLU
        assert isinstance(layer.act, nn.SiLU)

    def test_conv_no_act(self):
        from mmaivision.models.common import Conv
        layer = Conv(3, 16, k=1, s=1, act=False)
        out = layer(torch.randn(1, 3, 8, 8))
        assert out.shape == (1, 16, 8, 8)
        assert isinstance(layer.act, nn.Identity)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_models.py::TestConv -v`
Expected: FAIL with `ImportError` / `ModuleNotFoundError`

- [ ] **Step 3: 实现 autopad / make_divisible / Conv**

```python
# mmaivision/models/common.py
"""YOLOv5 共享算子,对齐 ultralytics/yolov5 models/common.py。"""
import math
from typing import Optional, Union, List

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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_models.py::TestConv -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add mmaivision/models/common.py tests/test_models.py
git commit -m "models: feat: 新增 YOLOv5 共享算子 Conv/autopad/make_divisible"
```

---

## Task 2: 实现 Bottleneck 和 C3

**Files:**
- Modify: `mmaivision/models/common.py`
- Modify: `tests/test_models.py`

- [ ] **Step 1: 追加 test_c3 测试**

```python
# tests/test_models.py — 追加在 TestConv 之后
class TestC3:
    def test_c3_shape_shortcut(self):
        from mmaivision.models.common import C3
        layer = C3(64, 64, n=2, shortcut=True)
        out = layer(torch.randn(1, 64, 32, 32))
        assert out.shape == (1, 64, 32, 32)

    def test_c3_shape_no_shortcut(self):
        from mmaivision.models.common import C3
        layer = C3(32, 64, n=1, shortcut=False)
        out = layer(torch.randn(1, 32, 16, 16))
        assert out.shape == (1, 64, 16, 16)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_models.py::TestC3 -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: 在 common.py 追加 Bottleneck 和 C3**

```python
# mmaivision/models/common.py — 在 Conv 之后追加
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_models.py::TestC3 -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add mmaivision/models/common.py tests/test_models.py
git commit -m "models: feat: 新增 YOLOv5 Bottleneck 和 C3 模块"
```

---

## Task 3: 实现 SPPF

**Files:**
- Modify: `mmaivision/models/common.py`
- Modify: `tests/test_models.py`

- [ ] **Step 1: 追加 test_sppf 测试**

```python
# tests/test_models.py — 追加
class TestSPPF:
    def test_sppf_shape(self):
        from mmaivision.models.common import SPPF
        layer = SPPF(64, 64, k=5)
        out = layer(torch.randn(1, 64, 32, 32))
        assert out.shape == (1, 64, 32, 32)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_models.py::TestSPPF -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: 在 common.py 追加 SPPF**

```python
# mmaivision/models/common.py — 追加
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_models.py::TestSPPF -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add mmaivision/models/common.py tests/test_models.py
git commit -m "models: feat: 新增 YOLOv5 SPPF 快速空间金字塔池化"
```

---

## Task 4: 实现 YOLOv5CSPDarknet backbone(s 变体形状测试)

**Files:**
- Create: `mmaivision/models/backbone.py`
- Modify: `mmaivision/models/__init__.py`
- Modify: `mmaivision/registry.py`(无修改,确认 MODELS Registry 已含 `mmaivision.models`)
- Modify: `tests/test_models.py`

- [ ] **Step 1: 追加 test_backbone_forward_shapes_s 测试**

```python
# tests/test_models.py — 追加
class TestBackbone:
    def test_backbone_forward_shapes_s(self):
        """yolov5s: deepen=0.33, widen=0.5。"""
        from mmaivision.models.backbone import YOLOv5CSPDarknet
        bb = YOLOv5CSPDarknet(deepen_factor=0.33, widen_factor=0.5)
        feats = bb(torch.randn(2, 3, 640, 640))
        assert len(feats) == 3
        # 基础通道 [64,128,256,512,1024] * 0.5 -> make_divisible/8
        # P3=256*0.5=128, P4=512*0.5=256, P5=1024*0.5=512
        assert feats[0].shape == (2, 128, 80, 80)   # P3, stride 8
        assert feats[1].shape == (2, 256, 40, 40)   # P4, stride 16
        assert feats[2].shape == (2, 512, 20, 20)   # P5, stride 32
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_models.py::TestBackbone::test_backbone_forward_shapes_s -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: 实现 backbone.py**

```python
# mmaivision/models/backbone.py
"""YOLOv5 CSPDarknet backbone。"""
from typing import Sequence, Tuple

import torch
from mmengine.model import BaseModule
from torch import Tensor, nn

from mmaivision.registry import MODELS
from .common import C3, Conv, SPPF, make_divisible


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
```

- [ ] **Step 4: 更新 __init__.py 触发注册**

```python
# mmaivision/models/__init__.py
from .backbone import YOLOv5CSPDarknet
from .model import CustomModel
from .weight_init import WEIGHT_INITIALIZERS
from .wrappers import CustomWrapper

__all__ = [
    'CustomModel', 'WEIGHT_INITIALIZERS', 'CustomWrapper',
    'YOLOv5CSPDarknet',
]
```

- [ ] **Step 5: 运行测试确认通过**

Run: `pytest tests/test_models.py::TestBackbone::test_backbone_forward_shapes_s -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add mmaivision/models/backbone.py mmaivision/models/__init__.py tests/test_models.py
git commit -m "models: feat: 新增 YOLOv5CSPDarknet backbone 输出 P3/P4/P5"
```

---

## Task 5: backbone 全家族变体测试 + 非法参数测试

**Files:**
- Modify: `tests/test_models.py`

- [ ] **Step 1: 追加全家族 + 非法参数测试**

```python
# tests/test_models.py — 追加在 TestBackbone 类内
    @pytest.mark.parametrize('variant,deepen,widen,expected_c', [
        ('n', 0.33, 0.25, (64, 128, 256)),
        ('s', 0.33, 0.50, (128, 256, 512)),
        ('m', 0.67, 0.75, (192, 384, 768)),
        ('l', 1.00, 1.00, (256, 512, 1024)),
        ('x', 1.33, 1.25, (320, 640, 1280)),
    ])
    def test_backbone_all_variants(self, variant, deepen, widen, expected_c):
        from mmaivision.models.backbone import YOLOv5CSPDarknet
        bb = YOLOv5CSPDarknet(deepen_factor=deepen, widen_factor=widen)
        feats = bb(torch.randn(1, 3, 320, 320))
        assert feats[0].shape[1] == expected_c[0], f'{variant} P3 channels'
        assert feats[1].shape[1] == expected_c[1], f'{variant} P4 channels'
        assert feats[2].shape[1] == expected_c[2], f'{variant} P5 channels'

    def test_backbone_invalid_factor_raises(self):
        from mmaivision.models.backbone import YOLOv5CSPDarknet
        with pytest.raises(ValueError, match='必须 > 0'):
            YOLOv5CSPDarknet(deepen_factor=0, widen_factor=0.5)
        with pytest.raises(ValueError, match='必须 > 0'):
            YOLOv5CSPDarknet(deepen_factor=0.33, widen_factor=-1)

    def test_backbone_invalid_out_indices_raises(self):
        from mmaivision.models.backbone import YOLOv5CSPDarknet
        with pytest.raises(ValueError, match='out_indices'):
            YOLOv5CSPDarknet(deepen_factor=0.33, widen_factor=0.5,
                             out_indices=(0, 1, 2))
```

- [ ] **Step 2: 运行测试确认通过**

Run: `pytest tests/test_models.py::TestBackbone -v`
Expected: PASS (1 + 5 参数化 + 2 = 8 tests)

注:n 变体通道是 `make_divisible(256*0.25, 8) = make_divisible(64) = 64`,m 变体是 `make_divisible(256*0.75) = make_divisible(192) = 192`。

- [ ] **Step 3: Commit**

```bash
git add tests/test_models.py
git commit -m "test: 新增 YOLOv5 backbone 全家族变体与非法参数测试"
```

---

## Task 6: 实现 YOLOv5PAFPN neck

**Files:**
- Create: `mmaivision/models/neck.py`
- Modify: `mmaivision/models/__init__.py`
- Modify: `tests/test_models.py`

- [ ] **Step 1: 追加 neck 测试**

```python
# tests/test_models.py — 追加
class TestNeck:
    def test_neck_forward_shapes(self):
        """s 变体 neck: 输入三层 (128,256,512) → 输出三层同 shape。"""
        from mmaivision.models.neck import YOLOv5PAFPN
        neck = YOLOv5PAFPN(
            in_channels=(128, 256, 512),
            out_channels=(128, 256, 512),
            deepen_factor=0.33, widen_factor=0.5)
        feats = (
            torch.randn(2, 128, 80, 80),
            torch.randn(2, 256, 40, 40),
            torch.randn(2, 512, 20, 20),
        )
        outs = neck(feats)
        assert len(outs) == 3
        assert outs[0].shape == (2, 128, 80, 80)
        assert outs[1].shape == (2, 256, 40, 40)
        assert outs[2].shape == (2, 512, 20, 20)

    def test_neck_wrong_in_channels_len_raises(self):
        from mmaivision.models.neck import YOLOv5PAFPN
        with pytest.raises(AssertionError):
            YOLOv5PAFPN(in_channels=(128, 256), out_channels=(128, 256),
                        deepen_factor=0.33, widen_factor=0.5)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_models.py::TestNeck -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: 实现 neck.py**

```python
# mmaivision/models/neck.py
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
```

- [ ] **Step 4: 更新 __init__.py**

```python
# mmaivision/models/__init__.py
from .backbone import YOLOv5CSPDarknet
from .model import CustomModel
from .neck import YOLOv5PAFPN
from .weight_init import WEIGHT_INITIALIZERS
from .wrappers import CustomWrapper

__all__ = [
    'CustomModel', 'WEIGHT_INITIALIZERS', 'CustomWrapper',
    'YOLOv5CSPDarknet', 'YOLOv5PAFPN',
]
```

- [ ] **Step 5: 运行测试确认通过**

Run: `pytest tests/test_models.py::TestNeck -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add mmaivision/models/neck.py mmaivision/models/__init__.py tests/test_models.py
git commit -m "models: feat: 新增 YOLOv5PAFPN PAN-FPN neck"
```

---

## Task 7: 实现 YOLOv5Head

**Files:**
- Create: `mmaivision/models/head.py`
- Modify: `mmaivision/models/__init__.py`
- Modify: `tests/test_models.py`

- [ ] **Step 1: 追加 head 测试**

```python
# tests/test_models.py — 追加
class TestHead:
    def test_head_forward_shapes(self):
        from mmaivision.models.head import YOLOv5Head
        head = YOLOv5Head(
            num_classes=80,
            in_channels=(128, 256, 512),
            num_base_priors=3)
        feats = (
            torch.randn(2, 128, 80, 80),
            torch.randn(2, 256, 40, 40),
            torch.randn(2, 512, 20, 20),
        )
        outs = head(feats)
        assert len(outs) == 3
        # num_base_priors * (num_classes + 5) = 3 * 85 = 255
        assert outs[0].shape == (2, 255, 80, 80)
        assert outs[1].shape == (2, 255, 40, 40)
        assert outs[2].shape == (2, 255, 20, 20)

    def test_head_invalid_args_raises(self):
        from mmaivision.models.head import YOLOv5Head
        with pytest.raises(ValueError):
            YOLOv5Head(num_classes=0, in_channels=(128, 256, 512))
        with pytest.raises(ValueError):
            YOLOv5Head(num_classes=80, in_channels=(128, 256, 512),
                       num_base_priors=0)

    def test_head_wrong_in_channels_len_raises(self):
        from mmaivision.models.head import YOLOv5Head
        with pytest.raises(AssertionError):
            YOLOv5Head(num_classes=80, in_channels=(128, 256))
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_models.py::TestHead -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: 实现 head.py**

```python
# mmaivision/models/head.py
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
```

- [ ] **Step 4: 更新 __init__.py**

```python
# mmaivision/models/__init__.py
from .backbone import YOLOv5CSPDarknet
from .head import YOLOv5Head
from .model import CustomModel
from .neck import YOLOv5PAFPN
from .weight_init import WEIGHT_INITIALIZERS
from .wrappers import CustomWrapper

__all__ = [
    'CustomModel', 'WEIGHT_INITIALIZERS', 'CustomWrapper',
    'YOLOv5CSPDarknet', 'YOLOv5PAFPN', 'YOLOv5Head',
]
```

- [ ] **Step 5: 运行测试确认通过**

Run: `pytest tests/test_models.py::TestHead -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add mmaivision/models/head.py mmaivision/models/__init__.py tests/test_models.py
git commit -m "models: feat: 新增 YOLOv5Head 三层 1x1 Conv 检测头"
```

---

## Task 8: 实现 YOLOv5Detector wrapper

**Files:**
- Create: `mmaivision/models/detector.py`
- Modify: `mmaivision/models/__init__.py`
- Modify: `tests/test_models.py`

- [ ] **Step 1: 追加 detector 端到端测试**

```python
# tests/test_models.py — 追加
class TestDetector:
    def _build(self):
        from mmaivision.registry import MODELS
        return MODELS.build(dict(
            type='YOLOv5Detector',
            backbone=dict(type='YOLOv5CSPDarknet',
                          deepen_factor=0.33, widen_factor=0.5),
            neck=dict(type='YOLOv5PAFPN',
                      in_channels=(128, 256, 512),
                      out_channels=(128, 256, 512),
                      deepen_factor=0.33, widen_factor=0.5),
            head=dict(type='YOLOv5Head',
                      num_classes=80,
                      in_channels=(128, 256, 512)),
        ))

    def test_detector_tensor_mode_end_to_end(self):
        # 直接调 .forward 跳过 BaseModel.__call__ 的 data_preprocessor 包装,
        # 这一轮不验证 data_preprocessor 路径。
        model = self._build()
        preds = model.forward(torch.randn(2, 3, 640, 640), mode='tensor')
        assert len(preds) == 3
        assert preds[0].shape == (2, 255, 80, 80)
        assert preds[1].shape == (2, 255, 40, 40)
        assert preds[2].shape == (2, 255, 20, 20)

    def test_detector_loss_mode_raises(self):
        model = self._build()
        with pytest.raises(NotImplementedError, match='loss'):
            model.forward(torch.randn(1, 3, 640, 640), mode='loss')

    def test_detector_predict_mode_raises(self):
        model = self._build()
        with pytest.raises(NotImplementedError, match='predict'):
            model.forward(torch.randn(1, 3, 640, 640), mode='predict')
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_models.py::TestDetector -v`
Expected: FAIL with `KeyError: 'YOLOv5Detector is not in the model registry'` 或 `ImportError`

- [ ] **Step 3: 实现 detector.py**

```python
# mmaivision/models/detector.py
"""YOLOv5Detector:串联 backbone → neck → head 的网络容器。"""
from typing import Optional

from mmengine.model import BaseModel
from torch import Tensor

from mmaivision.registry import MODELS


@MODELS.register_module()
class YOLOv5Detector(BaseModel):
    """串联 backbone → neck → head 的网络容器。

    本轮只实现 mode='tensor' 分支,用于 forward 验证 / 推理 / 可视化中间特征。
    loss / predict 分支待后续 loss + postprocess 模块完成。
    """

    def __init__(self,
                 backbone: dict,
                 neck: dict,
                 head: dict,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck)
        self.bbox_head = MODELS.build(head)

    def forward(self,
                inputs: Tensor,
                data_samples=None,
                mode: str = 'tensor'):
        if mode == 'tensor':
            return self.bbox_head(self.neck(self.backbone(inputs)))
        raise NotImplementedError(
            f"mode={mode!r} 暂未实现,需等待 loss/postprocess 模块。"
            "本轮仅支持 mode='tensor'。")
```

- [ ] **Step 4: 更新 __init__.py**

```python
# mmaivision/models/__init__.py
from .backbone import YOLOv5CSPDarknet
from .detector import YOLOv5Detector
from .head import YOLOv5Head
from .model import CustomModel
from .neck import YOLOv5PAFPN
from .weight_init import WEIGHT_INITIALIZERS
from .wrappers import CustomWrapper

__all__ = [
    'CustomModel', 'WEIGHT_INITIALIZERS', 'CustomWrapper',
    'YOLOv5CSPDarknet', 'YOLOv5PAFPN', 'YOLOv5Head', 'YOLOv5Detector',
]
```

- [ ] **Step 5: 运行测试确认通过**

Run: `pytest tests/test_models.py::TestDetector -v`
Expected: PASS (3 tests)

- [ ] **Step 6: 运行所有测试**

Run: `pytest tests/test_models.py -v`
Expected: 全部 PASS(common 3 + backbone 8 + neck 2 + head 3 + detector 3 = 19 tests)

- [ ] **Step 7: Commit**

```bash
git add mmaivision/models/detector.py mmaivision/models/__init__.py tests/test_models.py
git commit -m "models: feat: 新增 YOLOv5Detector 串联 backbone/neck/head"
```

---

## 最终验证

- [ ] 在仓库根目录运行 `pytest tests/test_models.py -v`,确认 19 个 test 全绿
- [ ] 运行 `pytest tests/ -v`,确认旧的 `test_datasets.py` 没有被破坏
- [ ] `git log --oneline -10` 确认每个 commit 按模块前缀分开
