# YOLOv5 网络结构集成设计

**日期:** 2026-05-23
**作者:** zhanggong
**状态:** 待实现

## 1. 背景与目标

将 YOLOv5 的网络结构(backbone / neck / head)集成进 mmaivision,作为目标检测模型的基础结构。本轮**只交付网络结构本身**,不含 loss / anchor / assigner / postprocess / nms — 这些留给后续独立轮次。

参考实现风格对齐 [ultralytics/yolov5](https://github.com/ultralytics/yolov5) 的 `models/common.py` + `models/yolo.py`,以便未来加载官方 `.pt` 预训练权重时只需做 state_dict key 重命名,不需要重训。

### 设计原则

- 完全走 mmengine Config dict + Registry 体系,不引入 ultralytics 的 yaml 解析机制
- 参数名 / 默认值与 ultralytics 严格对齐(`deepen_factor` ≙ `depth_multiple`,`widen_factor` ≙ `width_multiple`)
- 通过两个浮点乘子 `deepen_factor` / `widen_factor` 参数化全家族 n/s/m/l/x,避免每个变体一份代码
- backbone / neck / head 三者在 config 层完全解耦(`in_channels` 显式声明,不内部反推),可独立替换

### 不在本轮范围

- loss 函数(YOLOv5Loss / IoU loss / objectness loss)
- anchor 生成与 prior assignment
- decode / NMS / postprocess
- 加载 ultralytics `.pt` 权重(state_dict key 重命名)
- 训练 pipeline(数据增强 Mosaic / MixUp 等)

## 2. 架构

### 文件结构

```
mmaivision/models/
├── __init__.py              # 触发注册
├── common.py                # 共享算子:autopad / Conv / Bottleneck / C3 / SPPF
├── backbone.py              # @MODELS YOLOv5CSPDarknet
├── neck.py                  # @MODELS YOLOv5PAFPN
├── head.py                  # @MODELS YOLOv5Head
├── detector.py              # @MODELS YOLOv5Detector(BaseModel) — 串联 wrapper
├── model.py                 # 保留现有空骨架(本轮不动)
├── weight_init.py           # 保留现有(本轮不动)
└── wrappers.py              # 保留现有(本轮不动)
```

### 依赖关系

- `backbone / neck / head` 都只 `from .common import ...`,三者之间互不依赖
- `detector` 通过 `MODELS.build(cfg)` 动态构建三者,只在运行时依赖
- Registry 用 `mmaivision.registry.MODELS`,所有模块用 `@MODELS.register_module()` 注册
- `__init__.py` 仅 import 触发注册,不做 re-export 业务逻辑

### 外部入口

配置文件里写 `dict(type='YOLOv5Detector', backbone=..., neck=..., head=...)`,通过 `MODELS.build(cfg)` 一次拿到完整模型。

## 3. 组件接口

### 3.1 `common.py` — 共享算子

```python
def autopad(k, p=None, d=1):
    """same-padding 计算,支持 dilation。"""

def make_divisible(x, divisor=8):
    """向上取到 divisor 的整数倍,与 ultralytics 一致。"""

class Conv(nn.Module):
    """Conv + BN + SiLU,YOLOv5 通用卷积块。"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True): ...

class Bottleneck(nn.Module):
    """1x1 + 3x3 + optional shortcut。"""
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5): ...

class C3(nn.Module):
    """3-Conv CSP Bottleneck,YOLOv5 的核心模块。"""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5): ...

class SPPF(nn.Module):
    """快速 SPP,3 次 5x5 maxpool 串行。"""
    def __init__(self, c1, c2, k=5): ...
```

参数名 / 默认值与 ultralytics `models/common.py` 完全一致。

### 3.2 `backbone.py` — YOLOv5CSPDarknet

```python
@MODELS.register_module()
class YOLOv5CSPDarknet(BaseModule):
    """YOLOv5 CSPDarknet,输出 P3/P4/P5 三层特征。

    Args:
        deepen_factor (float): C3 块数 n 的乘子(对应 ultralytics depth_multiple)。
        widen_factor (float): 通道数 c 的乘子(对应 ultralytics width_multiple)。
        out_indices (Sequence[int]): 取哪几个 stage 作为输出,默认 (2, 3, 4) → P3/P4/P5。
        norm_cfg / act_cfg: 预留扩展接口,本轮固定 BN + SiLU。
    """
    # 基础通道 [64,128,256,512,1024],n 配方 [3,6,9,3],SPPF 放在最后
    def forward(self, x) -> Tuple[Tensor, ...]:  # (P3, P4, P5)
```

n/s/m/l/x 变体配方:

| 变体 | deepen_factor | widen_factor |
|------|---------------|--------------|
| n | 0.33 | 0.25 |
| s | 0.33 | 0.50 |
| m | 0.67 | 0.75 |
| l | 1.00 | 1.00 |
| x | 1.33 | 1.25 |

通道数走 `make_divisible(c * widen_factor, 8)`(与 ultralytics 一致);
`n_i = max(round(base_n * deepen_factor), 1)`,base = `[3, 6, 9, 3]`。

### 3.3 `neck.py` — YOLOv5PAFPN

```python
@MODELS.register_module()
class YOLOv5PAFPN(BaseModule):
    """YOLOv5 PAN-FPN,top-down + bottom-up,输出与输入同层数。

    Args:
        in_channels (Sequence[int]): 三个输入通道,如 (256, 512, 1024)*widen_factor。
        out_channels (Sequence[int]): 三个输出通道,通常与 in_channels 一致。
        deepen_factor (float): C3 块数乘子,与 backbone 一致。
        widen_factor (float): 通道乘子,与 backbone 一致。
    """
    def forward(self, feats: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        # 输入 (P3, P4, P5),输出 (N3, N4, N5)
```

### 3.4 `head.py` — YOLOv5Head

```python
@MODELS.register_module()
class YOLOv5Head(BaseModule):
    """YOLOv5 Detect head 的纯网络部分,只做 3 个 1x1 Conv。

    Args:
        num_classes (int): 类别数 nc。
        in_channels (Sequence[int]): 三个输入通道,通常等于 neck.out_channels。
        num_base_priors (int): 每层 anchor 数,默认 3(YOLOv5 标配),决定输出通道。
        widen_factor (float): 仅用于校验 in_channels 是否合理(可选)。
    """
    def forward(self, feats: Tuple[Tensor, ...]) -> List[Tensor]:
        # 输出 3 个 [B, num_base_priors*(num_classes+5), H, W]
```

### 3.5 `detector.py` — YOLOv5Detector

```python
@MODELS.register_module()
class YOLOv5Detector(BaseModel):
    """串联 backbone → neck → head 的网络容器。

    本轮只实现 mode='tensor' 分支(forward 验证 / 推理 / 可视化中间特征)。
    loss / predict 分支待后续 loss + postprocess 模块完成。
    """
    def __init__(self,
                 backbone: dict,
                 neck: dict,
                 head: dict,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck)
        self.bbox_head = MODELS.build(head)

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if mode == 'tensor':
            return self.bbox_head(self.neck(self.backbone(inputs)))
        raise NotImplementedError(
            f"mode={mode!r} 暂未实现,需等待 loss/postprocess 模块。"
            "本轮仅支持 mode='tensor'。")
```

backbone / neck / head 三者继承 `mmengine.model.BaseModule`,detector 继承 `mmengine.model.BaseModel`。

## 4. 数据流

### 输入

固定假设:4D `Tensor`,形状 `[B, 3, H, W]`,`H` 和 `W` 必须是 32 的倍数(stride 32 约束)。常用 640×640。

### Backbone forward

```
input: [B, 3, H, W]
  ↓ stem (Conv 6x6 s=2)        → [B, 64w, H/2, W/2]
  ↓ stage1 (Conv s=2 + C3 ×n1) → [B, 128w, H/4, W/4]
  ↓ stage2 (Conv s=2 + C3 ×n2) → [B, 256w, H/8, W/8]   ← P3
  ↓ stage3 (Conv s=2 + C3 ×n3) → [B, 512w, H/16, W/16] ← P4
  ↓ stage4 (Conv s=2 + C3 ×n4 + SPPF) → [B, 1024w, H/32, W/32] ← P5
output: (P3, P4, P5)
```

其中 `w = widen_factor`(走 make_divisible/8)。

### Neck forward(PAN-FPN)

```
input: (P3, P4, P5)
  Top-down:
    P5 → Conv 1x1 → up×2 → cat(P4) → C3 → M4
    M4 → Conv 1x1 → up×2 → cat(P3) → C3 → N3
  Bottom-up:
    N3 → Conv 3x3 s=2 → cat(M4) → C3 → N4
    N4 → Conv 3x3 s=2 → cat(P5 reduced) → C3 → N5
output: (N3, N4, N5),与输入同 stride、同通道数
```

### Head forward

```
input: (N3, N4, N5)
  对每层独立做 1x1 Conv: c_in → num_base_priors * (num_classes + 5)
output: [
  [B, na*(nc+5), H/8,  W/8],
  [B, na*(nc+5), H/16, W/16],
  [B, na*(nc+5), H/32, W/32],
]
```

### 三件套 + Detector 串联示例

```python
model = MODELS.build(dict(
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
preds = model(torch.randn(2, 3, 640, 640), mode='tensor')
# len(preds) == 3, preds[0].shape == (2, 3*(80+5), 80, 80)
```

`in_channels` 是已经乘过 `widen_factor` 之后的通道数,由 config 显式声明,这样 backbone / neck / head 在 config 层完全解耦。

## 5. 错误处理

YOLOv5 网络结构本身是纯前向计算,只在 **构造时(__init__)** 做防御性检查,forward 不做(让 PyTorch 自己抛 shape mismatch,信息更具体)。

### Backbone

- `deepen_factor / widen_factor` 必须 > 0,否则 `ValueError`
- `out_indices` 必须是 `(2,3,4)` 子集,否则 `ValueError`(本轮固定只支持 P3/P4/P5)

### Neck

- `in_channels` 和 `out_channels` 长度必须都是 3,否则 `AssertionError`

### Head

- `in_channels` 长度必须是 3,否则 `AssertionError`
- `num_classes >= 1`,`num_base_priors >= 1`,否则 `ValueError`

### Detector

- `mode` 只接受 `'tensor'`,其他值 `raise NotImplementedError`

### 不做的事

- 不做跨模块通道维度自动校验(由 config 维护者负责)
- 不做 input shape 校验(PyTorch 自己会暴露)
- 不做 NaN/Inf 检测(由 mmengine hook 负责)

## 6. 测试

### 测试文件

```
tests/test_models.py
```

单文件即可,模块小、互相独立、无 fixture 复用。

### 测试用例(共 11 个)

**common.py:**

1. `test_conv_shape` — `Conv(3, 16, k=3, s=2)` 输入 `[1,3,64,64]` → `[1,16,32,32]`,且 BN 存在、激活是 SiLU
2. `test_c3_shape_shortcut` — `C3(64, 64, n=2, shortcut=True)` 输入 `[1,64,32,32]` → `[1,64,32,32]`
3. `test_sppf_shape` — `SPPF(64, 64, k=5)` 输入 `[1,64,32,32]` → `[1,64,32,32]`

**backbone.py:**

4. `test_backbone_forward_shapes_s` — yolov5s(0.33/0.50)输入 `[2,3,640,640]` → 三个输出 `[2,128,80,80] / [2,256,40,40] / [2,512,20,20]`
5. `test_backbone_all_variants` — 参数化测 n/s/m/l/x 五个变体的输出通道数与 ultralytics 官方一致(查表对照)
6. `test_backbone_invalid_factor_raises` — `deepen_factor=0` / `widen_factor=-1` → `ValueError`

**neck.py:**

7. `test_neck_forward_shapes` — 输入三层匹配 s 变体的 backbone 输出 → 输出三层 shape 与输入完全一致
8. `test_neck_wrong_in_channels_len_raises` — `in_channels=(128, 256)` → `AssertionError`

**head.py:**

9. `test_head_forward_shapes` — 输入三层 (128/256/512) feature + `num_classes=80` + `num_base_priors=3` → 三层 `[B, 255, H, W]`
10. `test_head_invalid_args_raises` — `num_classes=0` → `ValueError`

**detector.py:**

11. `test_detector_tensor_mode_end_to_end` — 完整 build + `mode='tensor'` forward,检查 3 个输出 shape;再调一次 `mode='loss'` → `NotImplementedError`

### 不测的东西

- 不测梯度反传(PyTorch 自带测试覆盖)
- 不测数值正确性(无参考输出对照,本轮不做)
- 不测 init_weights(用 mmengine 默认初始化,无自定义逻辑)
- 不测训练循环(无 loss,跑不了)
- 不测性能 / FLOPs(超出本轮范围)

### 验证

```bash
pytest tests/test_models.py -v
```

11 个 test 全绿即视为本轮交付完成。
