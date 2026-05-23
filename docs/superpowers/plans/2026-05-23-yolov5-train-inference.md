# YOLOv5 Train/Inference Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为已有的 YOLOv5 backbone/neck/head/detector 补齐 anchor / bbox_coder / assigner / CIoU loss / postprocess,打通 `YOLOv5Detector.forward(mode='loss')` 和 `mode='predict')` 路径,使用 mock data 端到端验证。

**Architecture:** 5 套独立可测组件 — `YOLOv5AnchorGenerator` / `YOLOv5BBoxCoder` / `YOLOv5BatchAssigner` 注册到 `TASK_UTILS`,`bbox_ciou` 是纯函数;`YOLOv5Head` 新增 `loss_by_feat()` 与 `predict_by_feat()` 方法把这些组件串起来;`SingleStageDetector.loss/predict` 委托给 head。风格对齐 ultralytics v6.x/v7.x,验证用 mock data(本轮不打通 dataset/transforms)。

**Tech Stack:** PyTorch, mmengine (BaseModule/BaseModel/Registry/InstanceData), torchvision.ops.batched_nms, pytest

参考设计:[docs/superpowers/specs/2026-05-23-yolov5-train-inference-design.md](../specs/2026-05-23-yolov5-train-inference-design.md)

---

## 文件结构

| 路径 | 职责 | 状态 |
|------|------|------|
| `mmaivision/registry.py` | 给 `TASK_UTILS` 加 `locations='mmaivision.models.yolov5.task_utils'` | 改 |
| `mmaivision/models/yolov5/task_utils/__init__.py` | 触发 task_utils 子模块注册 | 新 |
| `mmaivision/models/yolov5/task_utils/prior_generator.py` | `YOLOv5AnchorGenerator` | 新 |
| `mmaivision/models/yolov5/task_utils/bbox_coder.py` | `YOLOv5BBoxCoder` | 新 |
| `mmaivision/models/yolov5/task_utils/assigner.py` | `YOLOv5BatchAssigner` | 新 |
| `mmaivision/models/yolov5/iou_loss.py` | `bbox_ciou` 纯函数 | 新 |
| `mmaivision/models/yolov5/head.py` | 加 task_utils 参数 + `loss_by_feat` + `predict_by_feat` | 改 |
| `mmaivision/models/yolov5/__init__.py` | `from . import task_utils  # noqa` 触发注册 | 改 |
| `mmaivision/models/base/single_stage.py` | `loss/predict` 委托给 head | 改 |
| `tests/test_yolov5_train.py` | 18 个新测试 | 新 |
| `tests/test_models.py` | 删 2 个 raise 测试 + 更新 TestHead 3 个测试的 init 参数 | 改 |

---

## Task 1: TASK_UTILS Registry locations + task_utils 子包骨架

**Files:**
- Modify: `mmaivision/registry.py` (TASK_UTILS Registry)
- Create: `mmaivision/models/yolov5/task_utils/__init__.py`
- Modify: `mmaivision/models/yolov5/__init__.py`

- [ ] **Step 1: 改 registry.py 给 TASK_UTILS 加 locations**

把 `mmaivision/registry.py` 中现有的:

```python
# manage task-specific modules like anchor generators and box coders
TASK_UTILS = Registry('task util', parent=MMENGINE_TASK_UTILS)
```

改成:

```python
# manage task-specific modules like anchor generators and box coders
TASK_UTILS = Registry(
    'task util',
    parent=MMENGINE_TASK_UTILS,
    locations=['mmaivision.models.yolov5.task_utils'])
```

- [ ] **Step 2: 创建 task_utils/__init__.py**

```python
# mmaivision/models/yolov5/task_utils/__init__.py
"""YOLOv5 训练 / 推理用工具组件:anchor / bbox_coder / assigner。"""
```

(内容暂时是空 docstring,后续 task 实现各模块后会在此 import 触发注册。)

- [ ] **Step 3: 改 yolov5/__init__.py 触发 task_utils 注册**

把 `mmaivision/models/yolov5/__init__.py` 改成:

```python
"""YOLOv5 模型族:backbone / neck / head / detector + task_utils。"""
from . import task_utils  # noqa: F401  触发 task_utils 注册
from .backbone import YOLOv5CSPDarknet
from .detector import YOLOv5Detector
from .head import YOLOv5Head
from .neck import YOLOv5PAFPN

__all__ = [
    'YOLOv5CSPDarknet',
    'YOLOv5PAFPN',
    'YOLOv5Head',
    'YOLOv5Detector',
]
```

- [ ] **Step 4: 验证 import 链不破**

Run: `pytest tests/ -v 2>&1 | tail -5`
Expected: 34 passed(原有测试不受影响)

- [ ] **Step 5: Commit**

```bash
git add mmaivision/registry.py mmaivision/models/yolov5/__init__.py mmaivision/models/yolov5/task_utils/__init__.py
git commit -m "models: feat: 为 TASK_UTILS Registry 添加 yolov5/task_utils 注册路径"
```

---

## Task 2: YOLOv5AnchorGenerator

**Files:**
- Create: `mmaivision/models/yolov5/task_utils/prior_generator.py`
- Modify: `mmaivision/models/yolov5/task_utils/__init__.py`
- Create: `tests/test_yolov5_train.py`

- [ ] **Step 1: 创建测试文件 + 写 TestAnchorGenerator 三个测试**

```python
# tests/test_yolov5_train.py
"""Tests for YOLOv5 训练 / 推理 pipeline。"""
import pytest
import torch
from torch import nn


class TestAnchorGenerator:
    def _build(self):
        from mmaivision.models.yolov5.task_utils.prior_generator import (
            YOLOv5AnchorGenerator)
        return YOLOv5AnchorGenerator(
            base_sizes=[
                [(10, 13), (16, 30), (33, 23)],
                [(30, 61), (62, 45), (59, 119)],
                [(116, 90), (156, 198), (373, 326)],
            ],
            strides=[8, 16, 32],
        )

    def test_grid_priors_shape(self):
        gen = self._build()
        priors = gen.grid_priors([(80, 80), (40, 40), (20, 20)])
        assert len(priors) == 3
        assert priors[0].shape == (3, 80, 80, 2)
        assert priors[1].shape == (3, 40, 40, 2)
        assert priors[2].shape == (3, 20, 20, 2)
        # base_sizes[0][0] / strides[0] = (10/8, 13/8)
        assert torch.allclose(priors[0][0, 0, 0],
                              torch.tensor([10 / 8, 13 / 8]))

    def test_grid_xy_shape_and_values(self):
        gen = self._build()
        grids = gen.grid_xy([(4, 5), (2, 3), (1, 2)])
        assert len(grids) == 3
        # 第 0 层:(ny=4, nx=5, 2)
        assert grids[0].shape == (4, 5, 2)
        assert torch.equal(grids[0][0, 0], torch.tensor([0., 0.]))
        # nx-1 = 4, ny-1 = 3
        assert torch.equal(grids[0][3, 4], torch.tensor([4., 3.]))

    def test_invalid_strides_raises(self):
        from mmaivision.models.yolov5.task_utils.prior_generator import (
            YOLOv5AnchorGenerator)
        with pytest.raises(ValueError, match='stride'):
            YOLOv5AnchorGenerator(
                base_sizes=[[(10, 13), (16, 30), (33, 23)]],
                strides=[0])
```

- [ ] **Step 2: 跑测试看失败**

Run: `pytest tests/test_yolov5_train.py::TestAnchorGenerator -v`
Expected: FAIL with `ModuleNotFoundError` (prior_generator not yet created)

- [ ] **Step 3: 实现 prior_generator.py**

```python
# mmaivision/models/yolov5/task_utils/prior_generator.py
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
```

- [ ] **Step 4: 在 task_utils/__init__.py 触发注册**

```python
# mmaivision/models/yolov5/task_utils/__init__.py
"""YOLOv5 训练 / 推理用工具组件:anchor / bbox_coder / assigner。"""
from .prior_generator import YOLOv5AnchorGenerator

__all__ = ['YOLOv5AnchorGenerator']
```

- [ ] **Step 5: 跑测试看通过**

Run: `pytest tests/test_yolov5_train.py::TestAnchorGenerator -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add mmaivision/models/yolov5/task_utils/prior_generator.py mmaivision/models/yolov5/task_utils/__init__.py tests/test_yolov5_train.py
git commit -m "models: feat: 新增 YOLOv5AnchorGenerator anchor 与 grid 生成器"
```

---

## Task 3: YOLOv5BBoxCoder

**Files:**
- Create: `mmaivision/models/yolov5/task_utils/bbox_coder.py`
- Modify: `mmaivision/models/yolov5/task_utils/__init__.py`
- Modify: `tests/test_yolov5_train.py`

- [ ] **Step 1: 追加 TestBBoxCoder 三个测试**

```python
# tests/test_yolov5_train.py — 追加
class TestBBoxCoder:
    def _build(self):
        from mmaivision.models.yolov5.task_utils.bbox_coder import (
            YOLOv5BBoxCoder)
        return YOLOv5BBoxCoder()

    def test_decode_known_values(self):
        coder = self._build()
        # pred=zeros 时 sigmoid(0)=0.5
        # xy 中心 = (0.5*2 - 0.5 + 5) * 8 = 44
        # wh = (0.5*2)^2 * 4 * 8 = 32  (anchor=4 网格 = 32 像素)
        pred = torch.zeros(1, 1, 1, 1, 4)
        anchor = torch.tensor([[4., 4.]])               # (na=1, 2) 网格单位
        grid_xy = torch.tensor([[[5., 5.]]])            # (ny=1, nx=1, 2)
        out = coder.decode(pred, anchor, grid_xy, stride=8)
        # 返回 xyxy 像素:中心 (44, 44),wh (32, 32) -> xyxy (28, 28, 60, 60)
        assert out.shape == (1, 1, 1, 1, 4)
        assert torch.allclose(out[0, 0, 0, 0],
                              torch.tensor([28., 28., 60., 60.]))

    def test_encode_decode_roundtrip(self):
        coder = self._build()
        # 给定 cxcywh,encode 出 tx/ty/tw/th,再 decode 应该回到原 cxcywh
        gt_xywh = torch.tensor([[44., 44., 32., 32.]])
        matched_anchor = torch.tensor([[4., 4.]])
        matched_grid_xy = torch.tensor([[5., 5.]])
        targets = coder.encode(gt_xywh, matched_anchor, matched_grid_xy, 8)
        # 用 decode 单点路径反推:把 targets 重新塞回 pred shape
        pred = torch.zeros(1, 1, 1, 1, 4)
        pred[0, 0, 0, 0] = targets[0]
        # logit 形式的 tx/ty/tw/th 通过 sigmoid 重建
        # 注意:encode 出的是 logit 之前的值,即 inv-sigmoid 之后的
        # 用 decode 比较麻烦,直接验证 decode(encode) ≈ gt 即可
        decoded = coder.decode(pred, matched_anchor,
                               torch.tensor([[[5., 5.]]]), 8)
        # decoded 是 xyxy,转回 cxcywh 比较
        x1, y1, x2, y2 = decoded[0, 0, 0, 0].tolist()
        cxcywh = [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
        assert all(abs(a - b) < 1e-3
                   for a, b in zip(cxcywh, gt_xywh[0].tolist()))

    def test_decode_batched_shape(self):
        coder = self._build()
        pred = torch.randn(2, 3, 20, 20, 4)
        anchor = torch.tensor([[1., 1.], [2., 2.], [3., 3.]])
        grid_xy = torch.zeros(20, 20, 2)
        for y in range(20):
            for x in range(20):
                grid_xy[y, x] = torch.tensor([x, y], dtype=torch.float32)
        out = coder.decode(pred, anchor, grid_xy, stride=8)
        assert out.shape == (2, 3, 20, 20, 4)
```

- [ ] **Step 2: 跑测试看失败**

Run: `pytest tests/test_yolov5_train.py::TestBBoxCoder -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: 实现 bbox_coder.py**

```python
# mmaivision/models/yolov5/task_utils/bbox_coder.py
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
        # 中心点反推 sigmoid 前值
        # sig*2 - 0.5 = cx/stride - grid_x  →  sig = (cx/stride - grid_x + 0.5) / 2
        cx_grid = gt_xywh[:, 0] / stride - matched_grid_xy[:, 0]
        cy_grid = gt_xywh[:, 1] / stride - matched_grid_xy[:, 1]
        sig_x = (cx_grid + 0.5) / 2
        sig_y = (cy_grid + 0.5) / 2
        tx = torch.logit(sig_x.clamp(1e-6, 1 - 1e-6))
        ty = torch.logit(sig_y.clamp(1e-6, 1 - 1e-6))
        # wh 反推: (sig*2)^2 * anchor = wh/stride  →  sig = sqrt(wh/(stride*anchor)) / 2
        w_grid = gt_xywh[:, 2] / stride
        h_grid = gt_xywh[:, 3] / stride
        sig_w = torch.sqrt(w_grid / matched_anchor[:, 0]) / 2
        sig_h = torch.sqrt(h_grid / matched_anchor[:, 1]) / 2
        tw = torch.logit(sig_w.clamp(1e-6, 1 - 1e-6))
        th = torch.logit(sig_h.clamp(1e-6, 1 - 1e-6))
        return torch.stack([tx, ty, tw, th], dim=-1)
```

- [ ] **Step 4: 更新 task_utils/__init__.py**

```python
"""YOLOv5 训练 / 推理用工具组件:anchor / bbox_coder / assigner。"""
from .bbox_coder import YOLOv5BBoxCoder
from .prior_generator import YOLOv5AnchorGenerator

__all__ = ['YOLOv5AnchorGenerator', 'YOLOv5BBoxCoder']
```

- [ ] **Step 5: 跑测试看通过**

Run: `pytest tests/test_yolov5_train.py::TestBBoxCoder -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add mmaivision/models/yolov5/task_utils/bbox_coder.py mmaivision/models/yolov5/task_utils/__init__.py tests/test_yolov5_train.py
git commit -m "models: feat: 新增 YOLOv5BBoxCoder 编解码器"
```

---

## Task 4: YOLOv5BatchAssigner

**Files:**
- Create: `mmaivision/models/yolov5/task_utils/assigner.py`
- Modify: `mmaivision/models/yolov5/task_utils/__init__.py`
- Modify: `tests/test_yolov5_train.py`

- [ ] **Step 1: 追加 TestAssigner 三个测试**

```python
# tests/test_yolov5_train.py — 追加
class TestAssigner:
    def _build(self, num_classes=80):
        from mmaivision.models.yolov5.task_utils.assigner import (
            YOLOv5BatchAssigner)
        return YOLOv5BatchAssigner(num_classes=num_classes)

    def _anchors_per_layer(self):
        # s 变体的网格单位 anchor:base_sizes / strides
        # base=[(10,13),(16,30),(33,23)] / 8 ≈ [(1.25,1.625),(2,3.75),(4.125,2.875)]
        return [
            torch.tensor([[1.25, 1.625], [2., 3.75], [4.125, 2.875]]),
            torch.tensor([[1.875, 3.8125], [3.875, 2.8125], [3.6875, 7.4375]]),
            torch.tensor([[3.625, 2.8125], [4.875, 6.1875], [11.65625, 10.1875]]),
        ]

    def test_assigner_basic_match(self):
        from mmengine.structures import InstanceData
        assigner = self._build(num_classes=80)
        # 1 张图,1 个 GT,xywh = (200, 200, 80, 80) 像素
        # 在 stride=32 层,grid 单位 wh = (80/32, 80/32) = (2.5, 2.5)
        # 与 stride=32 层第 0 个 anchor (3.625, 2.8125) 比例 < 4,应该能匹配
        gt = InstanceData(
            bboxes=torch.tensor([[160., 160., 240., 240.]]),  # xyxy
            labels=torch.tensor([5]),
        )
        anchors = self._anchors_per_layer()
        assignments = assigner([gt], anchors,
                               featmap_sizes=[(80, 80), (40, 40), (20, 20)])
        assert len(assignments) == 3
        total = sum(a['img_idx'].numel() for a in assignments)
        assert total > 0, '应至少有一层匹配上'

    def test_assigner_3grid_expansion(self):
        from mmengine.structures import InstanceData
        assigner = self._build(num_classes=80)
        # GT 中心 (160, 160) 在 stride=32 层对应 grid (5.0, 5.0) 正中
        # 验证扩展逻辑下不会无脑扩到 9 格
        gt = InstanceData(
            bboxes=torch.tensor([[100., 100., 220., 220.]]),
            labels=torch.tensor([0]),
        )
        anchors = self._anchors_per_layer()
        assignments = assigner([gt], anchors,
                               featmap_sizes=[(80, 80), (40, 40), (20, 20)])
        # 对每层、每个 anchor 索引,扩展数应 ≤ 3
        for layer_idx, a in enumerate(assignments):
            if a['img_idx'].numel() == 0:
                continue
            for ai in a['anchor_idx'].unique():
                count = (a['anchor_idx'] == ai).sum().item()
                assert count <= 3, \
                    f'layer {layer_idx} anchor {ai}: ' \
                    f'扩展数 {count} > 3'

    def test_assigner_empty_batch_gt(self):
        from mmengine.structures import InstanceData
        assigner = self._build()
        empty = InstanceData(
            bboxes=torch.zeros(0, 4),
            labels=torch.zeros(0, dtype=torch.int64),
        )
        anchors = self._anchors_per_layer()
        assignments = assigner([empty, empty], anchors,
                               featmap_sizes=[(80, 80), (40, 40), (20, 20)])
        assert len(assignments) == 3
        for a in assignments:
            for key in ('img_idx', 'anchor_idx', 'grid_y', 'grid_x',
                        'gt_xy', 'gt_wh', 'gt_class'):
                assert key in a
                assert a[key].numel() == 0
```

- [ ] **Step 2: 跑测试看失败**

Run: `pytest tests/test_yolov5_train.py::TestAssigner -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: 实现 assigner.py(签名比 spec §3.3 多一个必填 strides 参数,assigner 需要 stride 把像素 cxcywh 转网格单位)**

```python
# mmaivision/models/yolov5/task_utils/assigner.py
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
            wh_ratio = gt_layer[..., 4:6] / anchors_l.view(na, 1, 2)  # (na, T, 2)
            max_ratio = torch.maximum(wh_ratio, 1.0 / wh_ratio).max(dim=-1).values
            keep = max_ratio < self.prior_match_thr  # (na, T)
            gt_kept = gt_layer[keep]  # (M0, 7)

            if gt_kept.shape[0] == 0:
                results.append(self._empty_dict(device))
                continue

            # 3-grid 扩展:对每个 kept gt,选 ≤2 个邻近 grid
            gxy = gt_kept[:, 2:4]  # cxcy 网格单位
            gxy_inv = torch.tensor([nx, ny], device=device, dtype=torch.float32) - gxy
            j_pos = ((gxy % 1 < self.near_neighbor_thr) & (gxy > 1)).T  # (2, M0)
            k_pos = ((gxy_inv % 1 < self.near_neighbor_thr) & (gxy_inv > 1)).T  # (2, M0)
            mask = torch.stack(
                [torch.ones_like(j_pos[0]), j_pos[0], j_pos[1],
                 k_pos[0], k_pos[1]], dim=0)  # (5, M0)
            gt_ext = gt_kept.repeat(5, 1, 1)[mask]  # (M, 7)
            off_ext = off[:, None, :].repeat(1, gt_kept.shape[0], 1)[mask]  # (M, 2)

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
```

- [ ] **Step 4: 更新 task_utils/__init__.py**

```python
"""YOLOv5 训练 / 推理用工具组件:anchor / bbox_coder / assigner。"""
from .assigner import YOLOv5BatchAssigner
from .bbox_coder import YOLOv5BBoxCoder
from .prior_generator import YOLOv5AnchorGenerator

__all__ = [
    'YOLOv5AnchorGenerator',
    'YOLOv5BBoxCoder',
    'YOLOv5BatchAssigner',
]
```

- [ ] **Step 5: 跑测试看通过**

Run: `pytest tests/test_yolov5_train.py::TestAssigner -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add mmaivision/models/yolov5/task_utils/assigner.py mmaivision/models/yolov5/task_utils/__init__.py tests/test_yolov5_train.py
git commit -m "models: feat: 新增 YOLOv5BatchAssigner shape-based 匹配器"
```

---

## Task 5: bbox_ciou 函数

**Files:**
- Create: `mmaivision/models/yolov5/iou_loss.py`
- Modify: `tests/test_yolov5_train.py`

- [ ] **Step 1: 追加 TestCIoU 两个测试**

```python
# tests/test_yolov5_train.py — 追加
class TestCIoU:
    def test_ciou_identical_boxes(self):
        from mmaivision.models.yolov5.iou_loss import bbox_ciou
        pred = torch.tensor([[10., 10., 50., 50.]])
        target = torch.tensor([[10., 10., 50., 50.]])
        ciou = bbox_ciou(pred, target)
        assert ciou.shape == (1,)
        assert torch.allclose(ciou, torch.tensor([1.0]), atol=1e-5)

    def test_ciou_disjoint_boxes(self):
        from mmaivision.models.yolov5.iou_loss import bbox_ciou
        pred = torch.tensor([[0., 0., 10., 10.]])
        target = torch.tensor([[100., 100., 110., 110.]])
        ciou = bbox_ciou(pred, target)
        # 不相交且距离远,CIoU 应 < 0(IoU=0 - distance_penalty - aspect_penalty)
        assert ciou.item() < 0
```

- [ ] **Step 2: 跑测试看失败**

Run: `pytest tests/test_yolov5_train.py::TestCIoU -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: 实现 iou_loss.py**

```python
# mmaivision/models/yolov5/iou_loss.py
"""YOLOv5 用 CIoU 实现(对齐 ultralytics)。"""
import math

import torch
from torch import Tensor


def bbox_ciou(pred_xyxy: Tensor,
              target_xyxy: Tensor,
              eps: float = 1e-7) -> Tensor:
    """逐对 CIoU。

    Args:
        pred_xyxy: ``(N, 4)`` xyxy。
        target_xyxy: ``(N, 4)`` xyxy。
        eps: 数值稳定项。

    Returns:
        ``(N,)`` CIoU,范围 [-1, 1]。
    """
    # IoU
    px1, py1, px2, py2 = pred_xyxy.unbind(-1)
    tx1, ty1, tx2, ty2 = target_xyxy.unbind(-1)
    inter_x1 = torch.maximum(px1, tx1)
    inter_y1 = torch.maximum(py1, ty1)
    inter_x2 = torch.minimum(px2, tx2)
    inter_y2 = torch.minimum(py2, ty2)
    inter = (inter_x2 - inter_x1).clamp(min=0) \
        * (inter_y2 - inter_y1).clamp(min=0)
    p_area = (px2 - px1) * (py2 - py1)
    t_area = (tx2 - tx1) * (ty2 - ty1)
    union = p_area + t_area - inter + eps
    iou = inter / union

    # 中心距离
    p_cx = (px1 + px2) / 2
    p_cy = (py1 + py2) / 2
    t_cx = (tx1 + tx2) / 2
    t_cy = (ty1 + ty2) / 2
    center_dist_sq = (p_cx - t_cx) ** 2 + (p_cy - t_cy) ** 2

    # 外接矩形对角线
    enc_x1 = torch.minimum(px1, tx1)
    enc_y1 = torch.minimum(py1, ty1)
    enc_x2 = torch.maximum(px2, tx2)
    enc_y2 = torch.maximum(py2, ty2)
    enc_diag_sq = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + eps

    # aspect ratio penalty
    pw = (px2 - px1).clamp(min=eps)
    ph = (py2 - py1).clamp(min=eps)
    tw = (tx2 - tx1).clamp(min=eps)
    th = (ty2 - ty1).clamp(min=eps)
    v = (4 / math.pi ** 2) * (torch.atan(tw / th) - torch.atan(pw / ph)) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    return iou - center_dist_sq / enc_diag_sq - alpha * v
```

- [ ] **Step 4: 跑测试看通过**

Run: `pytest tests/test_yolov5_train.py::TestCIoU -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add mmaivision/models/yolov5/iou_loss.py tests/test_yolov5_train.py
git commit -m "models: feat: 新增 bbox_ciou CIoU 函数"
```

---

## Task 6: YOLOv5Head 新构造参数 + 更新旧测试

**Files:**
- Modify: `mmaivision/models/yolov5/head.py`
- Modify: `tests/test_models.py`

新增 keyword-only 参数:`prior_generator / bbox_coder / assigner / strides / 6 个 loss 权重&postprocess 阈值`。同步更新 `tests/test_models.py::TestHead` 三个旧测试,加上 task_utils 配置。

- [ ] **Step 1: 改 head.py 新签名 + 构造校验**

把 `mmaivision/models/yolov5/head.py` 改成:

```python
"""YOLOv5 Detect head:三层 1x1 Conv + loss_by_feat + predict_by_feat。"""
from typing import List, Sequence, Tuple

import torch.nn as nn
from mmengine.model import BaseModule
from torch import Tensor

from mmaivision.registry import MODELS, TASK_UTILS


@MODELS.register_module()
class YOLOv5Head(BaseModule):
    """YOLOv5 Detect head。

    forward 输出三层原始 feature(本网络结构部分,不做 decode)。
    loss_by_feat / predict_by_feat 在后续 Task 实现。

    Args:
        num_classes: 类别数 nc。
        in_channels: 三个输入通道,通常等于 neck.out_channels。
        prior_generator: ``dict``,YOLOv5AnchorGenerator 配置。
        bbox_coder: ``dict``,YOLOv5BBoxCoder 配置。
        assigner: ``dict``,YOLOv5BatchAssigner 配置。
        num_base_priors: 每层 anchor 数,默认 3。
        strides: 三层 stride,默认 (8, 16, 32)。
        loss_box_weight / loss_obj_weight / loss_cls_weight: loss 权重。
        obj_level_weights: 三层 obj loss 缩放,默认 (4.0, 1.0, 0.4)。
        score_thr / nms_iou_thr / max_per_img: postprocess 阈值。
    """

    def __init__(self,
                 *,
                 num_classes: int,
                 in_channels: Sequence[int],
                 prior_generator: dict,
                 bbox_coder: dict,
                 assigner: dict,
                 num_base_priors: int = 3,
                 strides: Sequence[int] = (8, 16, 32),
                 loss_box_weight: float = 0.05,
                 loss_obj_weight: float = 1.0,
                 loss_cls_weight: float = 0.5,
                 obj_level_weights: Sequence[float] = (4.0, 1.0, 0.4),
                 score_thr: float = 0.001,
                 nms_iou_thr: float = 0.45,
                 max_per_img: int = 300,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        assert len(in_channels) == 3, \
            f'in_channels 长度必须为 3, got {len(in_channels)}'
        assert len(strides) == 3, \
            f'strides 长度必须为 3, got {len(strides)}'
        assert len(obj_level_weights) == 3, \
            f'obj_level_weights 长度必须为 3, got {len(obj_level_weights)}'
        if num_classes < 1:
            raise ValueError(f'num_classes 必须 >= 1, got {num_classes}')
        if num_base_priors < 1:
            raise ValueError(
                f'num_base_priors 必须 >= 1, got {num_base_priors}')
        if not (0 < score_thr < 1):
            raise ValueError(
                f'score_thr 必须 ∈ (0, 1), got {score_thr}')
        if not (0 < nms_iou_thr < 1):
            raise ValueError(
                f'nms_iou_thr 必须 ∈ (0, 1), got {nms_iou_thr}')
        if max_per_img < 1:
            raise ValueError(f'max_per_img 必须 >= 1, got {max_per_img}')
        for w_name, w in [('loss_box_weight', loss_box_weight),
                          ('loss_obj_weight', loss_obj_weight),
                          ('loss_cls_weight', loss_cls_weight)]:
            if w < 0:
                raise ValueError(f'{w_name} 必须 >= 0, got {w}')

        self.num_classes = num_classes
        self.num_base_priors = num_base_priors
        self.strides = list(strides)
        self.loss_box_weight = loss_box_weight
        self.loss_obj_weight = loss_obj_weight
        self.loss_cls_weight = loss_cls_weight
        self.obj_level_weights = list(obj_level_weights)
        self.score_thr = score_thr
        self.nms_iou_thr = nms_iou_thr
        self.max_per_img = max_per_img

        self.prior_generator = TASK_UTILS.build(prior_generator)
        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.assigner = TASK_UTILS.build(assigner)

        out_c = num_base_priors * (num_classes + 5)
        self.convs = nn.ModuleList(
            [nn.Conv2d(c, out_c, kernel_size=1) for c in in_channels])

    def forward(self, feats: Tuple[Tensor, ...]) -> List[Tensor]:
        return [conv(f) for conv, f in zip(self.convs, feats)]
```

- [ ] **Step 2: 更新 tests/test_models.py 中 TestHead 三个测试,补 task_utils 配置**

把 `tests/test_models.py` 中 TestHead 类改成:

```python
class TestHead:
    def _task_utils_cfg(self):
        return dict(
            prior_generator=dict(
                type='YOLOv5AnchorGenerator',
                base_sizes=[
                    [(10, 13), (16, 30), (33, 23)],
                    [(30, 61), (62, 45), (59, 119)],
                    [(116, 90), (156, 198), (373, 326)],
                ],
                strides=[8, 16, 32]),
            bbox_coder=dict(type='YOLOv5BBoxCoder'),
            assigner=dict(
                type='YOLOv5BatchAssigner',
                num_classes=80,
                strides=[8, 16, 32]),
        )

    def test_head_forward_shapes(self):
        from mmaivision.models.yolov5.head import YOLOv5Head
        import torch
        head = YOLOv5Head(
            num_classes=80,
            in_channels=(128, 256, 512),
            num_base_priors=3,
            **self._task_utils_cfg())
        feats = (
            torch.randn(2, 128, 80, 80),
            torch.randn(2, 256, 40, 40),
            torch.randn(2, 512, 20, 20),
        )
        outs = head(feats)
        assert len(outs) == 3
        assert outs[0].shape == (2, 255, 80, 80)
        assert outs[1].shape == (2, 255, 40, 40)
        assert outs[2].shape == (2, 255, 20, 20)

    def test_head_invalid_args_raises(self):
        from mmaivision.models.yolov5.head import YOLOv5Head
        import pytest
        cfg = self._task_utils_cfg()
        # 把 assigner num_classes 也降到 1,否则 num_classes=0 先过不了 assigner build
        cfg['assigner']['num_classes'] = 1
        with pytest.raises(ValueError):
            YOLOv5Head(num_classes=0, in_channels=(128, 256, 512), **cfg)
        with pytest.raises(ValueError):
            YOLOv5Head(num_classes=80, in_channels=(128, 256, 512),
                       num_base_priors=0, **self._task_utils_cfg())

    def test_head_wrong_in_channels_len_raises(self):
        from mmaivision.models.yolov5.head import YOLOv5Head
        import pytest
        with pytest.raises(AssertionError):
            YOLOv5Head(num_classes=80, in_channels=(128, 256),
                       **self._task_utils_cfg())
```

注:`TestDetector::_build` 也用到 head config(`mmaivision/models/yolov5/detector.py` 的测试),需要同步更新 — 加 task_utils 配置:

```python
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
                      in_channels=(128, 256, 512),
                      prior_generator=dict(
                          type='YOLOv5AnchorGenerator',
                          base_sizes=[
                              [(10, 13), (16, 30), (33, 23)],
                              [(30, 61), (62, 45), (59, 119)],
                              [(116, 90), (156, 198), (373, 326)],
                          ],
                          strides=[8, 16, 32]),
                      bbox_coder=dict(type='YOLOv5BBoxCoder'),
                      assigner=dict(
                          type='YOLOv5BatchAssigner',
                          num_classes=80,
                          strides=[8, 16, 32]),
                      ),
        ))
    # ...test_detector_tensor_mode_end_to_end 保留
    # test_detector_loss_mode_raises 和 test_detector_predict_mode_raises
    # 保留不动(下个 task 才会删,本 task 不动)
```

- [ ] **Step 3: 跑测试看通过**

Run: `pytest tests/test_models.py -v 2>&1 | tail -10`
Expected: 21 passed(全部旧测试,新接口下仍跑得通)

- [ ] **Step 4: Commit**

```bash
git add mmaivision/models/yolov5/head.py tests/test_models.py
git commit -m "models: feat: 扩展 YOLOv5Head 构造参数为 task_utils 与 loss/postprocess 配置"
```

---

## Task 7: YOLOv5Head.loss_by_feat

**Files:**
- Modify: `mmaivision/models/yolov5/head.py`
- Modify: `tests/test_yolov5_train.py`

- [ ] **Step 1: 追加 TestHeadLossByFeat 三个测试**

```python
# tests/test_yolov5_train.py — 追加
class TestHeadLossByFeat:
    def _make_head(self, num_classes=80):
        from mmaivision.models.yolov5.head import YOLOv5Head
        return YOLOv5Head(
            num_classes=num_classes,
            in_channels=(128, 256, 512),
            prior_generator=dict(
                type='YOLOv5AnchorGenerator',
                base_sizes=[
                    [(10, 13), (16, 30), (33, 23)],
                    [(30, 61), (62, 45), (59, 119)],
                    [(116, 90), (156, 198), (373, 326)],
                ],
                strides=[8, 16, 32]),
            bbox_coder=dict(type='YOLOv5BBoxCoder'),
            assigner=dict(
                type='YOLOv5BatchAssigner',
                num_classes=num_classes,
                strides=[8, 16, 32]),
        )

    def _make_pred_maps(self, B=2, nc=80, na=3):
        return [
            torch.randn(B, na * (nc + 5), 80, 80, requires_grad=True),
            torch.randn(B, na * (nc + 5), 40, 40, requires_grad=True),
            torch.randn(B, na * (nc + 5), 20, 20, requires_grad=True),
        ]

    def test_loss_by_feat_basic(self):
        from mmengine.structures import InstanceData
        head = self._make_head()
        pred_maps = self._make_pred_maps()
        batch_gt = [
            InstanceData(
                bboxes=torch.tensor([[10., 20., 100., 200.],
                                     [50., 50., 150., 250.]]),
                labels=torch.tensor([0, 1])),
            InstanceData(
                bboxes=torch.tensor([[200., 300., 400., 500.]]),
                labels=torch.tensor([2])),
        ]
        metas = [dict(batch_input_shape=(640, 640))] * 2
        losses = head.loss_by_feat(pred_maps, batch_gt, metas)
        assert 'loss_bbox' in losses
        assert 'loss_obj' in losses
        assert 'loss_cls' in losses
        for v in losses.values():
            assert torch.isfinite(v).all(), f'loss should be finite, got {v}'
            assert v.item() >= 0
        total = sum(losses.values())
        total.backward()  # 不报错即可
        # pred_maps 第一层应收到非零梯度
        assert pred_maps[0].grad is not None
        assert pred_maps[0].grad.abs().sum().item() > 0

    def test_loss_by_feat_all_empty_gt(self):
        from mmengine.structures import InstanceData
        head = self._make_head()
        pred_maps = self._make_pred_maps()
        empty = InstanceData(
            bboxes=torch.zeros(0, 4),
            labels=torch.zeros(0, dtype=torch.int64))
        metas = [dict(batch_input_shape=(640, 640))] * 2
        losses = head.loss_by_feat(pred_maps, [empty, empty], metas)
        assert losses['loss_bbox'].item() == 0.0
        assert losses['loss_cls'].item() == 0.0
        assert losses['loss_obj'].item() > 0
        sum(losses.values()).backward()

    def test_loss_by_feat_batch_mismatch_raises(self):
        from mmengine.structures import InstanceData
        head = self._make_head()
        pred_maps = self._make_pred_maps(B=2)
        # 故意只给 1 个 gt(B=2)
        batch_gt = [InstanceData(bboxes=torch.zeros(0, 4),
                                  labels=torch.zeros(0, dtype=torch.int64))]
        metas = [dict(batch_input_shape=(640, 640))]
        with pytest.raises(AssertionError):
            head.loss_by_feat(pred_maps, batch_gt, metas)
```

- [ ] **Step 2: 跑测试看失败**

Run: `pytest tests/test_yolov5_train.py::TestHeadLossByFeat -v`
Expected: FAIL (loss_by_feat not yet implemented, returns None / AttributeError)

- [ ] **Step 3: 在 head.py 追加 loss_by_feat**

在 `mmaivision/models/yolov5/head.py` 末尾(forward 方法之后)追加:

```python
    def loss_by_feat(self,
                     pred_maps: List[Tensor],
                     batch_gt_instances,
                     batch_img_metas) -> dict:
        """计算三段 loss(bbox CIoU + obj BCE + cls BCE)。

        Args:
            pred_maps: 三层 ``[B, na*(nc+5), Hi, Wi]``。
            batch_gt_instances: 长度 B 的 InstanceData 列表。
            batch_img_metas: 长度 B 的 dict 列表(目前不读字段,占位)。

        Returns:
            dict 包含 ``loss_bbox / loss_obj / loss_cls``,已乘权重。
        """
        from .iou_loss import bbox_ciou
        import torch
        import torch.nn.functional as F

        B = pred_maps[0].shape[0]
        assert len(batch_gt_instances) == B, \
            f'batch_gt_instances 长度 {len(batch_gt_instances)} ' \
            f'!= pred batch {B}'

        nc = self.num_classes
        na = self.num_base_priors
        device = pred_maps[0].device
        dtype = pred_maps[0].dtype
        del batch_img_metas  # 当前实现不需要 metas

        featmap_sizes = [(p.shape[2], p.shape[3]) for p in pred_maps]
        anchors_per_layer = self.prior_generator.grid_priors(
            featmap_sizes, device=device, dtype=dtype)
        # 上面返回 (na, ny, nx, 2);assigner 期望 (na, 2),取 [: ,0, 0]
        anchors_for_assigner = [a[:, 0, 0, :] for a in anchors_per_layer]
        assignments = self.assigner(
            batch_gt_instances, anchors_for_assigner, featmap_sizes)

        loss_bbox = torch.zeros((), device=device, dtype=dtype)
        loss_obj = torch.zeros((), device=device, dtype=dtype)
        loss_cls = torch.zeros((), device=device, dtype=dtype)

        bce = F.binary_cross_entropy_with_logits

        for i, raw_map in enumerate(pred_maps):
            Hi, Wi = raw_map.shape[2], raw_map.shape[3]
            raw = raw_map.view(B, na, nc + 5, Hi, Wi).permute(
                0, 1, 3, 4, 2).contiguous()  # (B, na, Hi, Wi, nc+5)

            a = assignments[i]
            M = a['img_idx'].numel()
            obj_target = torch.zeros((B, na, Hi, Wi), device=device, dtype=dtype)

            if M > 0:
                matched_anchor = anchors_for_assigner[i][a['anchor_idx']]
                pos = raw[a['img_idx'], a['anchor_idx'], a['grid_y'], a['grid_x']]
                pos_xy = pos[:, 0:2].sigmoid() * 2 - 0.5
                pos_wh = (pos[:, 2:4].sigmoid() * 2) ** 2 * matched_anchor
                pos_obj = pos[:, 4]
                pos_cls = pos[:, 5:]

                grid_xy_int = torch.stack(
                    [a['grid_x'].float(), a['grid_y'].float()], dim=-1)
                pred_cxcywh = torch.cat([pos_xy, pos_wh], dim=-1)
                target_cxcywh = torch.cat(
                    [a['gt_xy'] - grid_xy_int, a['gt_wh']], dim=-1)
                pred_xyxy = _cxcywh_to_xyxy(pred_cxcywh)
                target_xyxy = _cxcywh_to_xyxy(target_cxcywh)
                ciou = bbox_ciou(pred_xyxy, target_xyxy)
                loss_bbox = loss_bbox + (1.0 - ciou).mean()

                obj_target[a['img_idx'], a['anchor_idx'],
                           a['grid_y'], a['grid_x']] = ciou.detach().clamp(0)

                if nc > 1:
                    cls_target = F.one_hot(
                        a['gt_class'], num_classes=nc).to(dtype)
                    loss_cls = loss_cls + bce(pos_cls, cls_target).mean()

            loss_obj_layer = bce(raw[..., 4], obj_target).mean()
            loss_obj = loss_obj + loss_obj_layer * self.obj_level_weights[i]

        loss_bbox = loss_bbox * self.loss_box_weight * B
        loss_obj = loss_obj * self.loss_obj_weight * B
        loss_cls = loss_cls * self.loss_cls_weight * B

        return dict(loss_bbox=loss_bbox, loss_obj=loss_obj, loss_cls=loss_cls)


def _cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack(
        [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
```

注意:把 `_cxcywh_to_xyxy` 写在文件 module 级末尾。`import torch` 在 head.py 顶部需要补上(原 head.py 只 import 了 `torch.nn as nn`)— 加 `import torch`。

- [ ] **Step 4: 跑测试看通过**

Run: `pytest tests/test_yolov5_train.py::TestHeadLossByFeat -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add mmaivision/models/yolov5/head.py tests/test_yolov5_train.py
git commit -m "models: feat: 新增 YOLOv5Head.loss_by_feat 三段 loss 实现"
```

---

## Task 8: YOLOv5Head.predict_by_feat

**Files:**
- Modify: `mmaivision/models/yolov5/head.py`
- Modify: `tests/test_yolov5_train.py`

- [ ] **Step 1: 追加 TestHeadPredictByFeat**

```python
# tests/test_yolov5_train.py — 追加
class TestHeadPredictByFeat:
    def _make_head(self, num_classes=80):
        from mmaivision.models.yolov5.head import YOLOv5Head
        return YOLOv5Head(
            num_classes=num_classes,
            in_channels=(128, 256, 512),
            prior_generator=dict(
                type='YOLOv5AnchorGenerator',
                base_sizes=[
                    [(10, 13), (16, 30), (33, 23)],
                    [(30, 61), (62, 45), (59, 119)],
                    [(116, 90), (156, 198), (373, 326)],
                ],
                strides=[8, 16, 32]),
            bbox_coder=dict(type='YOLOv5BBoxCoder'),
            assigner=dict(
                type='YOLOv5BatchAssigner',
                num_classes=num_classes,
                strides=[8, 16, 32]),
            score_thr=0.01,
            nms_iou_thr=0.45,
        )

    def test_predict_by_feat_returns_instancedata(self):
        from mmengine.structures import InstanceData
        head = self._make_head()
        B = 2
        pred_maps = [
            torch.randn(B, 3 * 85, 80, 80),
            torch.randn(B, 3 * 85, 40, 40),
            torch.randn(B, 3 * 85, 20, 20),
        ]
        metas = [dict(batch_input_shape=(640, 640))] * B
        preds = head.predict_by_feat(pred_maps, metas)
        assert isinstance(preds, list)
        assert len(preds) == B
        for p in preds:
            assert isinstance(p, InstanceData)
            assert hasattr(p, 'bboxes')
            assert hasattr(p, 'scores')
            assert hasattr(p, 'labels')
            assert p.bboxes.ndim == 2 and p.bboxes.shape[1] == 4
            assert p.scores.ndim == 1
            assert p.labels.ndim == 1 and p.labels.dtype == torch.int64
            assert p.bboxes.shape[0] == p.scores.shape[0] == p.labels.shape[0]
```

- [ ] **Step 2: 跑测试看失败**

Run: `pytest tests/test_yolov5_train.py::TestHeadPredictByFeat -v`
Expected: FAIL (predict_by_feat not implemented)

- [ ] **Step 3: 在 head.py 追加 predict_by_feat**

在 `mmaivision/models/yolov5/head.py` 的 `loss_by_feat` 之后追加:

```python
    def predict_by_feat(self,
                        pred_maps: List[Tensor],
                        batch_img_metas) -> list:
        """解码 + per-image NMS,返回 List[InstanceData]。"""
        import torch
        from mmengine.structures import InstanceData
        from torchvision.ops import batched_nms

        B = pred_maps[0].shape[0]
        nc = self.num_classes
        na = self.num_base_priors
        device = pred_maps[0].device
        dtype = pred_maps[0].dtype
        del batch_img_metas  # 当前实现不读 metas

        featmap_sizes = [(p.shape[2], p.shape[3]) for p in pred_maps]
        anchors = self.prior_generator.grid_priors(
            featmap_sizes, device=device, dtype=dtype)
        grid_xy = self.prior_generator.grid_xy(
            featmap_sizes, device=device, dtype=dtype)

        # 各层解码后拼成 (B, total_N, nc+5)
        all_xyxy = []
        all_scores = []  # (B, total_N, nc)

        for i, raw_map in enumerate(pred_maps):
            Hi, Wi = raw_map.shape[2], raw_map.shape[3]
            stride = self.strides[i]
            raw = raw_map.view(B, na, nc + 5, Hi, Wi).permute(
                0, 1, 3, 4, 2).contiguous()  # (B, na, Hi, Wi, nc+5)
            sig = torch.sigmoid(raw)
            xy = (sig[..., 0:2] * 2 - 0.5 + grid_xy[i].view(1, 1, Hi, Wi, 2)) \
                * stride
            wh = (sig[..., 2:4] * 2) ** 2 \
                * anchors[i].view(1, na, Hi, Wi, 2) * stride
            obj = sig[..., 4:5]
            cls = sig[..., 5:]
            score = obj * cls  # (B, na, Hi, Wi, nc)

            x1y1 = xy - wh / 2
            x2y2 = xy + wh / 2
            xyxy = torch.cat([x1y1, x2y2], dim=-1)  # (B, na, Hi, Wi, 4)

            # flatten (na, Hi, Wi) → N
            xyxy = xyxy.view(B, -1, 4)
            score = score.view(B, -1, nc)
            all_xyxy.append(xyxy)
            all_scores.append(score)

        all_xyxy = torch.cat(all_xyxy, dim=1)    # (B, N, 4)
        all_scores = torch.cat(all_scores, dim=1)  # (B, N, nc)

        results = []
        for b in range(B):
            scores_b, labels_b = all_scores[b].max(dim=-1)
            keep = scores_b > self.score_thr
            bboxes_kept = all_xyxy[b][keep]
            scores_kept = scores_b[keep]
            labels_kept = labels_b[keep]
            if bboxes_kept.shape[0] == 0:
                results.append(InstanceData(
                    bboxes=torch.zeros(0, 4, device=device, dtype=dtype),
                    scores=torch.zeros(0, device=device, dtype=dtype),
                    labels=torch.zeros(0, dtype=torch.int64, device=device)))
                continue
            keep_idx = batched_nms(
                bboxes_kept, scores_kept, labels_kept, self.nms_iou_thr)
            keep_idx = keep_idx[:self.max_per_img]
            results.append(InstanceData(
                bboxes=bboxes_kept[keep_idx],
                scores=scores_kept[keep_idx],
                labels=labels_kept[keep_idx]))
        return results
```

- [ ] **Step 4: 跑测试看通过**

Run: `pytest tests/test_yolov5_train.py::TestHeadPredictByFeat -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add mmaivision/models/yolov5/head.py tests/test_yolov5_train.py
git commit -m "models: feat: 新增 YOLOv5Head.predict_by_feat decode+NMS"
```

---

## Task 9: SingleStageDetector 委托 + 删旧 raise 测试 + 端到端测试

**Files:**
- Modify: `mmaivision/models/base/single_stage.py`
- Modify: `tests/test_models.py`
- Modify: `tests/test_yolov5_train.py`

- [ ] **Step 1: 改 single_stage.py 让 loss/predict 委托给 head**

把 `mmaivision/models/base/single_stage.py` 末尾两个方法改成:

```python
    def loss(self, inputs: Tensor, data_samples=None):
        feats = self.extract_feat(inputs)
        pred_maps = self.bbox_head(feats)
        batch_gt = [s.gt_instances for s in data_samples]
        batch_metas = [s.metainfo for s in data_samples]
        return self.bbox_head.loss_by_feat(pred_maps, batch_gt, batch_metas)

    def predict(self, inputs: Tensor, data_samples=None):
        feats = self.extract_feat(inputs)
        pred_maps = self.bbox_head(feats)
        if data_samples is not None:
            batch_metas = [s.metainfo for s in data_samples]
        else:
            batch_metas = [dict(batch_input_shape=tuple(inputs.shape[-2:]))
                           ] * inputs.shape[0]
        return self.bbox_head.predict_by_feat(pred_maps, batch_metas)
```

- [ ] **Step 2: 删 tests/test_models.py 中两个旧 raise 测试**

把 `TestDetector` 中以下两个方法删掉:

```python
    def test_detector_loss_mode_raises(self):
        ...
    def test_detector_predict_mode_raises(self):
        ...
```

保留 `_build` 和 `test_detector_tensor_mode_end_to_end`。

- [ ] **Step 3: 追加 TestDetectorEndToEnd 三个测试到 tests/test_yolov5_train.py**

```python
# tests/test_yolov5_train.py — 追加
class TestDetectorEndToEnd:
    def _build_model(self):
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
                      in_channels=(128, 256, 512),
                      prior_generator=dict(
                          type='YOLOv5AnchorGenerator',
                          base_sizes=[
                              [(10, 13), (16, 30), (33, 23)],
                              [(30, 61), (62, 45), (59, 119)],
                              [(116, 90), (156, 198), (373, 326)],
                          ],
                          strides=[8, 16, 32]),
                      bbox_coder=dict(type='YOLOv5BBoxCoder'),
                      assigner=dict(
                          type='YOLOv5BatchAssigner',
                          num_classes=80,
                          strides=[8, 16, 32])),
        ))

    def _make_data(self, B=2):
        from mmengine.structures import BaseDataElement, InstanceData
        inputs = torch.randn(B, 3, 640, 640)
        samples = [
            BaseDataElement(
                gt_instances=InstanceData(
                    bboxes=torch.tensor([[10., 20., 100., 200.],
                                         [50., 50., 150., 250.]]),
                    labels=torch.tensor([0, 1])),
                metainfo=dict(batch_input_shape=(640, 640))),
            BaseDataElement(
                gt_instances=InstanceData(
                    bboxes=torch.tensor([[200., 300., 400., 500.]]),
                    labels=torch.tensor([2])),
                metainfo=dict(batch_input_shape=(640, 640))),
        ]
        return inputs, samples[:B]

    def test_detector_loss_mode_runs(self):
        model = self._build_model()
        inputs, samples = self._make_data(B=2)
        losses = model.forward(inputs, samples, mode='loss')
        assert 'loss_bbox' in losses
        assert 'loss_obj' in losses
        assert 'loss_cls' in losses
        total = sum(losses.values())
        total.backward()
        # backbone stem.conv 应收到非零梯度
        stem_grad = model.backbone.stem.conv.weight.grad
        assert stem_grad is not None
        assert stem_grad.abs().sum().item() > 0

    def test_detector_predict_mode_returns_instancedata(self):
        from mmengine.structures import InstanceData
        model = self._build_model()
        inputs, samples = self._make_data(B=2)
        preds = model.forward(inputs, samples, mode='predict')
        assert isinstance(preds, list)
        assert len(preds) == 2
        for p in preds:
            assert isinstance(p, InstanceData)

    def test_detector_unknown_mode_raises(self):
        model = self._build_model()
        inputs, samples = self._make_data(B=1)
        with pytest.raises(ValueError, match='mode'):
            model.forward(inputs, samples, mode='wrong')
```

- [ ] **Step 4: 跑全量测试**

Run: `pytest tests/ -v 2>&1 | tail -10`
Expected: 全部 PASS。总数大致 = 34(原)- 2(删) + 18(新增 yolov5_train) = **50 passed**(可能因 BBoxCoder roundtrip 测试和 assigner 内部细节有小幅波动,以最终实测为准)

- [ ] **Step 5: Commit**

```bash
git add mmaivision/models/base/single_stage.py tests/test_models.py tests/test_yolov5_train.py
git commit -m "models: feat: SingleStageDetector.loss/predict 委托给 head 并补端到端测试"
```

---

## 最终验证

- [ ] 在仓库根运行 `pytest tests/ -v`,所有测试通过(约 50 个)
- [ ] `git log --oneline -10` 验证 9 个 commit 按 task 分开,中文 + 模块前缀
- [ ] `MODELS.build` 整套 YOLOv5Detector(含 head 的 task_utils)能成功
- [ ] `model.forward(inputs, data_samples, mode='loss')` 返回 dict 三个 loss,`.backward()` 不报错
- [ ] `model.forward(inputs, data_samples, mode='predict')` 返回 List[InstanceData]
