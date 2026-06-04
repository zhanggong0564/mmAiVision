# YOLOv5 实例分割 (YOLOv5-seg) 设计

**日期:** 2026-06-04
**作者:** zhanggong
**状态:** 待实现
**前置:** 依赖 [2026-05-23-yolov5-train-inference-design.md](2026-05-23-yolov5-train-inference-design.md)、[2026-05-22-labelme-det-dataset-design.md](2026-05-22-labelme-det-dataset-design.md)

## 1. 背景与目标

现有 YOLOv5 检测链路完整(数据 → 模型 → loss → NMS → mAP 评估 → demo),但 labelme/X-AnyLabeling 标注里的 **polygon 轮廓信息被丢弃**:

- [datasets.py:162-164](../../../mmaivision/datasets/datasets.py#L162) 已把 polygon 解析为 `inst['mask']`,但下游全程不用;
- [transforms.py:38-52](../../../mmaivision/datasets/transforms.py#L38) 的 `LoadLabelmeAnnotations` 只取 bbox;
- `LetterResize` / `PackDetInputs` 不处理 mask;
- `YOLOv5Head` 无 mask 分支;`LabelmeDetMetric` 只算检测 mAP。

**目标:** 在保留检测能力的前提下,补齐 **实例分割(instance segmentation)** 全链路,使每个 `line` / `QFU` 目标输出独立 mask,能训练、能评估(mask mAP)、能推理可视化。

风格对齐 [ultralytics/yolov5 segment](https://github.com/ultralytics/yolov5)(YOLACT 风格:共享 proto 原型 + per-anchor mask 系数),但完全用 mmengine 体系实现,不引入 ultralytics 代码依赖。

### 范围(本轮做)

全链路一次做完:数据 transform、Proto 模块、seg head(loss + predict)、assigner 加 `gt_idx`、seg 检测器、mask mAP 指标、demo 可视化、convert 工具加 seg 支持、config、测试。

### 关键设计原则

1. **新增类,不改检测原类。** `YOLOv5SegHead` / `YOLOv5SegDetector` / `LabelmeSegMetric` 均为新类;`YOLOv5Head` / `YOLOv5Detector` / `LabelmeDetMetric` 零改动。检测 config 不受影响。
2. **assigner 纯加列。** `YOLOv5BatchAssigner` 输出新增 `gt_idx`(正样本对应的全局 gt 实例索引),现有 `YOLOv5Head` 不读它 → 检测训练行为完全不变。这是唯一被触碰的检测期文件,且为向后兼容的加法改动。
3. **忠实对齐官方。** gt mask 在输入分辨率(640)栅格化,loss 内下采样到 proto 分辨率(160);mask loss 用 bbox 裁剪 + 面积归一。便于从 `yolov5n-seg.pt` warm-start。

### 不在本轮范围

- Mosaic / MixUp / copy-paste 等带 mask 的数据增强
- mask 的 autoanchor / EMA / multi-scale
- COCO 风格 segm AP(用自包含 VOC 风格 mask mAP)

## 2. 架构

### 文件结构

```
mmaivision/
├── datasets/
│   └── transforms.py                 # 改:Load/LetterResize/Pack 加 polygon→mask
├── models/
│   └── yolov5/
│       ├── common.py                 # 改:新增 Proto 模块
│       ├── head.py                   # 改:新增 YOLOv5SegHead 类(原类不动)
│       ├── detector.py               # 改:新增 YOLOv5SegDetector 类(原类不动)
│       └── task_utils/
│           └── assigner.py           # 改:输出加 gt_idx 列(加法,行为不变)
├── evaluation/
│   └── metrics.py                    # 改:新增 LabelmeSegMetric 类(原类不动)
tools/
└── convert_ultralytics.py            # 改:PREFIX_MAP 加 proto,支持 nm
demo/
└── yolov5_seg_demo.py                # 新:mask 叠加可视化
configs/
└── yolov5_n_seg_labelme.py           # 新:seg 训练配置
tests/
└── test_seg.py                       # 新:seg 链路测试
```

### 核心常量

- `nm = 32`:mask 系数 / proto 通道数(官方默认)。
- `npr = 256`:proto 中间通道(官方默认)。
- proto 分辨率 = 输入 / 4(P3 stride 8 上采样 ×2),640 → 160×160。
- head 每 anchor 输出维度 `no = nc + 5 + nm`;conv 输出通道 `na * no`。

## 3. 逐模块设计

### 3.1 数据 transform ([transforms.py](../../../mmaivision/datasets/transforms.py))

**`LoadLabelmeAnnotations`**(改):
- 在现有 bbox 输出基础上,从 `inst['mask']`(polygon flat 点列)构造 `results['gt_polygons']`:`List[np.ndarray(K,2) float32]`,长度 = 实例数。
- 无 polygon 的实例(纯 rectangle 标注)给空多边形占位 → 栅格化后为全 0 mask,该实例不参与 mask loss(bbox loss 照常)。

**`LetterResize`**(改):
- 已有 bbox `×r + pad`;同步对每个 polygon 点 `×r`、`x += left`、`y += top`,并 clip 到 `[0, scale]`。

**`PackDetInputs`**(改):
- 用 `cv2.fillPoly` 把每个 polygon 栅格化到 `(scale, scale)` uint8,堆成 `(N, H, W)` tensor 写入 `gt_instances.masks`。
- N=0 时写 `(0, H, W)` 空 tensor。
- 检测路径不受影响(`masks` 是新增字段)。

### 3.2 Proto 模块 ([common.py](../../../mmaivision/models/yolov5/common.py))

```
Proto(c1, npr=256, nm=32):
  cv1 = Conv(c1, npr, k=3)
  upsample = nn.Upsample(scale_factor=2, mode='nearest')
  cv2 = Conv(npr, npr, k=3)
  cv3 = Conv(npr, nm, k=1)        # 官方 Proto.cv3 = Conv(c_, nm) 默认 1x1
  forward(x): cv3(cv2(upsample(cv1(x))))   # (B, nm, 2H, 2W)
```
复用本仓库已有 `Conv`(与 ultralytics 命名一致),保证 convert 时子模块 key 对齐。

### 3.3 seg head ([head.py](../../../mmaivision/models/yolov5/head.py))

新增 `YOLOv5SegHead(YOLOv5Head)`:
- `__init__`:加 `num_masks=32` / `proto_channels=256`;`out_c = na*(nc+5+nm)`;构建 `self.proto = Proto(in_channels[0], proto_channels, num_masks)`。其余复用父类(anchor / coder / assigner)。
- `forward(feats)`:返回 `(pred_maps, proto)`,`proto = self.proto(feats[0])`(P3)。
- `loss_by_feat(pred_maps, proto, batch_gt_instances, batch_img_metas)`:
  - 检测三段 loss 完全复用父类逻辑(可抽公共方法或内部复用)。
  - **mask loss**:对每层正样本,用 `gt_idx` 取该正样本预测的 32 维系数 → `coeffs @ proto[img]`(160×160) → sigmoid → 取对应 gt mask 下采样到 160 → 按 gt bbox(缩放到 160)裁剪 → BCE → 除以 bbox 面积归一 → batch 内求和。乘 `loss_mask_weight`。
  - 返回 `dict(loss_bbox, loss_obj, loss_cls, loss_mask)`。
- `predict_by_feat(pred_maps, proto, batch_img_metas)`:
  - 复用父类 decode + NMS 得到每图保留框 / score / label / **对应系数**(系数随框一起 gather)。
  - mask:`coeffs @ proto[b]` → sigmoid → 裁剪到框 → 上采样回输入分辨率 → 阈值 0.5 → `InstanceData.masks`(bool `(K, H, W)`)。

### 3.4 assigner ([assigner.py](../../../mmaivision/models/yolov5/task_utils/assigner.py))

- 拼 `all_gt` 时追加一列「全局 gt 索引」(0..T-1)。
- 输出 dict 增加 `gt_idx`(`int64`);`_empty_dict` 同步加空 `gt_idx`。
- 现有 `YOLOv5Head.loss_by_feat` 不读 `gt_idx` → 检测行为零变化(已有 test 应仍全绿)。

### 3.5 seg 检测器 ([detector.py](../../../mmaivision/models/yolov5/detector.py))

新增 `YOLOv5SegDetector(SingleStageDetector)`:
- `loss`:`feats → (pred_maps, proto) = head(feats)`;`head.loss_by_feat(pred_maps, proto, batch_gt, batch_metas)`。
- `predict`:`(pred_maps, proto) = head(feats)`;`head.predict_by_feat(...)`;把含 `masks` 的 `InstanceData` 挂回 `data_sample.pred_instances`(沿用现有约定)。
- `forward(mode='tensor')`:返回 `head(feats)`(tuple)。

### 3.6 mask mAP 指标 ([metrics.py](../../../mmaivision/evaluation/metrics.py))

新增 `LabelmeSegMetric(BaseMetric)`,`default_prefix='labelme_seg'`:
- 复用 `_voc_ap` 与 TP/FP 累积流程(可把通用 AP 逻辑抽成模块级函数共享)。
- `process`:存 pred/gt 的 `masks`(bool array)、scores、labels。
- IoU:`mask_iou = (a & b).sum / (a | b).sum`,替换 bbox IoU。
- 输出 `mAP` / `mAP_50` / `AP50_<cls>`(键加 `seg` 前缀区分检测)。
- pred 与 gt mask 都在 letterbox 输入坐标系(640)比较,保证对齐。

### 3.7 demo (`demo/yolov5_seg_demo.py`)

- 复用现有推理脚手架(参照 [yolov5_inference.py](../../../demo/yolov5_inference.py));
- 在画框基础上,对每个实例 mask 上色 + 半透明 alpha 叠加;保存可视化图。

### 3.8 convert 工具 ([convert_ultralytics.py](../../../tools/convert_ultralytics.py))

- `build_target_model`:加 `--seg` 开关,seg 时构建 `YOLOv5SegDetector` + `YOLOv5SegHead`(`num_classes=80, nm=32`)以逐层对齐 `yolov5n-seg.pt`。
- `PREFIX_MAP`:seg 模式追加
  - `bbox_head.proto.cv1 → model.24.proto.cv1` 等 proto 子模块;
  - `model.24.m.*` 通道因 `+nm` 变化,用同样 num_classes+nm 构建即可逐层等价。
- verify:seg 模式比对 detect 分支 raw 输出 + proto 输出。
- **双支持**:seg config 的 `load_from` 用 seg 权重(strict 命中);若只有检测权重,config 仍能 `strict=False` 加载(backbone/neck warm-start,proto/系数从零训)。

### 3.9 config (`configs/yolov5_n_seg_labelme.py`)

- 基于 `yolov5_n_labelme.py`,改:`YOLOv5SegDetector` / `YOLOv5SegHead`(加 `num_masks`、`loss_mask_weight`)、pipeline 不变(transform 自动产出 mask)、`val_evaluator=LabelmeSegMetric`、`save_best='labelme_seg/mAP_50'`。
- `load_from` 指向(可选)转换出的 `yolov5n-seg` 权重,缺省则复用检测权重。
- 原 `yolov5_n_labelme.py` 不动。

## 4. 数据流

```
labelme polygon ──Load──► gt_polygons ──LetterResize(×r+pad)──► Pack(fillPoly)
                                                                    │
                                                          gt_instances.masks (N,640,640) uint8
                                                                    │
img → backbone → neck ─P3─► Proto ───────────────► proto (32,160,160)
                      └───► SegHead convs ───────► pred_maps (na*(nc+5+32))
  训练: assigner(gt_idx) → 正样本系数 @ proto → 裁剪 → vs 下采样 gt mask → BCE/面积归一
  推理: NMS(框+系数) → 系数 @ proto → 裁剪/上采样/阈值0.5 → InstanceData.masks
  评估: mask IoU → VOC 风格 mask mAP
```

## 5. 错误处理与边界

- 实例无 polygon(纯 rectangle):空 mask,mask loss 跳过该实例,bbox loss 照常。
- 整图无正样本:`loss_mask = 0`(保持 graph 连通,proto 仍前向)。
- 退化 polygon(点数 < 3):`fillPoly` 得空 mask,等同无 polygon。
- 推理 0 检测:`masks` 为 `(0, H, W)`。

## 6. 测试 (`tests/test_seg.py`)

- Proto 输出形状 `(B,32,160,160)`。
- `YOLOv5SegHead.forward` 返回 `(list, tensor)`;`loss_by_feat` 返回 4 个 loss 且可 backward;`predict_by_feat` 产出 `masks`。
- assigner 输出含 `gt_idx`,且与 `gt_class` 长度一致;**回归:现有检测 test 全绿**。
- polygon → mask 栅格化正确性(已知三角形面积近似)。
- mask IoU / mask mAP 在构造样本上数值正确。
- (可选, 需网络)convert seg 权重数值校验 PASS。

## 7. 提交拆分(对齐 CLAUDE.md 模块规范)

按模块分多次提交,沿用本仓库历史前缀(`models` / `datasets` / `config` / `evaluation` / `tools` / `demo` / `test`)。预计:
- `models: feat: 新增 Proto 模块`
- `models: feat: 新增 YOLOv5SegHead(loss/predict 含 mask)`
- `models: refactor: assigner 输出加 gt_idx`
- `models: feat: 新增 YOLOv5SegDetector`
- `datasets: feat: transform 支持 polygon→mask`
- `evaluation: feat: 新增 LabelmeSegMetric mask mAP`
- `tools: feat: convert 支持 yolov5-seg 权重`
- `demo: feat: 新增 seg 可视化 demo`
- `config: feat: 新增 yolov5_n_seg_labelme 配置`
- `test: feat: 新增 seg 链路测试`
