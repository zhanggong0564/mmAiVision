# YOLOv5 训练与推理 Pipeline 设计

**日期:** 2026-05-23
**作者:** zhanggong
**状态:** 待实现
**前置:** 依赖 [2026-05-23-yolov5-network-design.md](2026-05-23-yolov5-network-design.md)

## 1. 背景与目标

为已实现的 YOLOv5 backbone / neck / head / detector(commit `66bdd48`)补齐 **训练 + 推理** 链路:实现 anchor 生成 / bbox 编解码 / shape-based assigner / 三段 loss(CIoU + obj BCE + cls BCE) / postprocess(decode + NMS),并将 `YOLOv5Detector.forward(mode='loss')` 和 `mode='predict')` 真正打通,使其不再 `raise NotImplementedError`。

风格对齐 [ultralytics/yolov5 v6.x/v7.x](https://github.com/ultralytics/yolov5) 的 `ComputeLoss` + `non_max_suppression`,但完全用 mmengine 体系(Registry / BaseModel / InstanceData),不引入 ultralytics 代码依赖。

### 范围(本轮做)

- 5 套新组件:`YOLOv5AnchorGenerator` / `YOLOv5BBoxCoder` / `YOLOv5BatchAssigner` / `bbox_ciou` 函数 / `YOLOv5Head.loss_by_feat + predict_by_feat`
- `SingleStageDetector.loss/predict` 改为委托给 head
- `mmaivision/registry.py` 为 `TASK_UTILS` 加 `locations`
- 端到端测试用 mock data(手工构造 `inputs + List[BaseDataElement]`)验证

### 不在本轮范围(留下一轮)

- 加载 ultralytics `.pt` 权重(独立子项目 B)
- 数据 pipeline:transforms / data_preprocessor / collate / Mosaic / MixUp
- autoanchor(无论是 CLI 脚本还是 train-time hook)
- 与 ultralytics 数值对齐的 unit-level 验证(等下一轮 .pt 权重加载后做端到端对齐)
- EMA / multi-scale 训练
- 与真实 dataset / evaluation metric 的对接

## 2. 架构

### 文件结构

```
mmaivision/
├── registry.py                       # 改:TASK_UTILS 加 locations
└── models/
    ├── base/
    │   └── single_stage.py           # 改:loss/predict 委托给 head
    └── yolov5/
        ├── common.py                 # 不动
        ├── backbone.py               # 不动
        ├── neck.py                   # 不动
        ├── head.py                   # 改:新增 loss_by_feat + predict_by_feat,
        │                             #     __init__ 增加 prior_generator/bbox_coder/
        │                             #     assigner/strides/loss 权重/postprocess 阈值
        ├── detector.py               # 不动(继续复用基类)
        ├── iou_loss.py               # 新:bbox_ciou 函数
        ├── __init__.py               # 改:可能 re-export task_utils 子包
        └── task_utils/               # 新子包
            ├── __init__.py
            ├── prior_generator.py    # @TASK_UTILS YOLOv5AnchorGenerator
            ├── bbox_coder.py         # @TASK_UTILS YOLOv5BBoxCoder
            └── assigner.py           # @TASK_UTILS YOLOv5BatchAssigner

tests/
├── test_models.py                    # 改:更新 TestDetector 中 2 个 raise 测试
└── test_yolov5_train.py              # 新:18 个新测试
```

### Registry 使用

- `MODELS`:已有的 backbone/neck/head/detector,本轮不增
- `TASK_UTILS`:新增 `YOLOv5AnchorGenerator` / `YOLOv5BBoxCoder` / `YOLOv5BatchAssigner`。需要给 `registry.py` 加 `locations=['mmaivision.models.yolov5.task_utils']`(以后加 yolov8/task_utils 时再追加一项)
- 不引入 LOSS_MODULES Registry — 本轮 loss 计算内联在 `head.loss_by_feat`,IoU 是普通函数

### 依赖关系

```
detector → base/single_stage → head
head → task_utils.{prior_generator, bbox_coder, assigner} + iou_loss
```

### 变更影响面

- 改动 5 个文件:`registry.py` / `base/single_stage.py` / `yolov5/head.py` / `yolov5/__init__.py` / `tests/test_models.py`
- 新增 6 个文件:`iou_loss.py` / `task_utils/{__init__.py, prior_generator.py, bbox_coder.py, assigner.py}` / `tests/test_yolov5_train.py`

## 3. 组件接口

### 3.1 `task_utils/prior_generator.py` — YOLOv5AnchorGenerator

```python
@TASK_UTILS.register_module()
class YOLOv5AnchorGenerator:
    """YOLOv5 anchor 生成:固定 anchors + 按 featmap_size 生成 grid。"""

    def __init__(self,
                 base_sizes: Sequence[Sequence[Tuple[int, int]]],
                 strides: Sequence[int]):
        # base_sizes[i] = [(w0,h0),(w1,h1),(w2,h2)] 第 i 层 3 个 anchor (像素单位)
        # strides[i]    = 该层 stride (8/16/32)
        ...

    @property
    def num_levels(self) -> int: ...      # 通常 3
    @property
    def num_base_priors(self) -> List[int]: ...   # [3, 3, 3]

    def grid_priors(self,
                    featmap_sizes: Sequence[Tuple[int, int]],
                    device='cpu',
                    dtype=torch.float32) -> List[Tensor]:
        # 每层返回 (na, ny, nx, 2) 的 anchor wh,单位:网格(stride 单位)
        # 即 base_sizes[i] / strides[i]
        ...

    def grid_xy(self,
                featmap_sizes: Sequence[Tuple[int, int]],
                device='cpu',
                dtype=torch.float32) -> List[Tensor]:
        # 每层返回 (ny, nx, 2) 的 grid 中心坐标(网格单位)
        ...
```

### 3.2 `task_utils/bbox_coder.py` — YOLOv5BBoxCoder

```python
@TASK_UTILS.register_module()
class YOLOv5BBoxCoder:
    """YOLOv5 编解码 box (中心点 sigmoid*2-0.5,宽高 (sigmoid*2)^2 * anchor)。"""

    def decode(self,
               pred: Tensor,            # (B, na, ny, nx, 4) 已 reshape
               anchor: Tensor,          # (na, 2) 网格单位
               grid_xy: Tensor,         # (ny, nx, 2) 网格单位
               stride: int) -> Tensor:  # 返回 xyxy,像素单位
        ...

    def encode(self,
               gt_xywh: Tensor,         # (M, 4) 像素单位 cxcywh
               matched_anchor: Tensor,  # (M, 2)
               matched_grid_xy: Tensor, # (M, 2)
               stride: int) -> Tensor:  # 返回 tx,ty,tw,th targets
        ...
```

注:loss 计算中实际不调 `encode`(YOLOv5 直接对 raw pred 做 sigmoid 再算 IoU,不走 tx/ty 路径)。`encode()` 作为对外可用接口保留,本轮被测但未被 loss 调用。

### 3.3 `task_utils/assigner.py` — YOLOv5BatchAssigner

```python
@TASK_UTILS.register_module()
class YOLOv5BatchAssigner:
    """YOLOv5 shape-based + 邻近 grid 扩展(3-grid)的 batch 匹配。

    实现对齐 ultralytics ComputeLoss.build_targets。
    """

    def __init__(self,
                 num_classes: int,
                 num_base_priors: int = 3,
                 prior_match_thr: float = 4.0,
                 near_neighbor_thr: float = 0.5):
        ...

    def __call__(self,
                 batch_gt_instances: List[InstanceData],
                 anchors: List[Tensor],           # 每层 (na, 2),网格单位
                 featmap_sizes: List[Tuple[int, int]]
                 ) -> List[Dict[str, Tensor]]:
        # 每层返回 dict:
        #   img_idx:    (M,) 图片索引
        #   anchor_idx: (M,) anchor 索引 [0..na-1]
        #   grid_y:     (M,)
        #   grid_x:     (M,)
        #   gt_xy:      (M, 2) cxcy 网格单位
        #   gt_wh:      (M, 2) wh 网格单位
        #   gt_class:   (M,) class index
        ...
```

### 3.4 `iou_loss.py` — CIoU 工具函数

```python
def bbox_ciou(pred_xyxy: Tensor, target_xyxy: Tensor, eps: float = 1e-7) -> Tensor:
    """逐对 CIoU,返回 (N,) 范围 [-1, 1] 的 CIoU(用 1-ciou 作 loss)。"""
    ...
```

不注册,纯函数。

### 3.5 `yolov5/head.py` — 新增方法

```python
@MODELS.register_module()
class YOLOv5Head(BaseModule):
    def __init__(self, *,
                 num_classes,
                 in_channels,
                 num_base_priors=3,
                 prior_generator: dict,               # 新增
                 bbox_coder: dict,                    # 新增
                 assigner: dict,                      # 新增
                 strides: Sequence[int] = (8, 16, 32),  # 新增
                 # loss 权重(对齐 ultralytics v6 默认)
                 loss_box_weight: float = 0.05,
                 loss_obj_weight: float = 1.0,
                 loss_cls_weight: float = 0.5,
                 obj_level_weights: Sequence[float] = (4.0, 1.0, 0.4),
                 # postprocess 默认
                 score_thr: float = 0.001,
                 nms_iou_thr: float = 0.45,
                 max_per_img: int = 300,
                 init_cfg=None):
        ...
        self.prior_generator = TASK_UTILS.build(prior_generator)
        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.assigner = TASK_UTILS.build(assigner)

    def forward(self, feats): ...   # 已有,不动

    def loss_by_feat(self,
                     pred_maps: List[Tensor],            # 三层 [B, na*(nc+5), H, W]
                     batch_gt_instances: List[InstanceData],
                     batch_img_metas: List[dict]
                     ) -> Dict[str, Tensor]:
        """返回 dict(loss_bbox, loss_obj, loss_cls),已乘权重和 batch_size 缩放。"""

    def predict_by_feat(self,
                        pred_maps: List[Tensor],
                        batch_img_metas: List[dict]
                        ) -> List[InstanceData]:
        """返回 List[InstanceData(bboxes=[N,4] xyxy, scores=[N], labels=[N])]"""
```

### 3.6 `base/single_stage.py` — 改 loss/predict

```python
class SingleStageDetector(BaseModel):
    # ...existing...

    def loss(self, inputs, data_samples):
        feats = self.extract_feat(inputs)
        pred_maps = self.bbox_head(feats)
        batch_gt = [s.gt_instances for s in data_samples]
        batch_metas = [s.metainfo for s in data_samples]
        return self.bbox_head.loss_by_feat(pred_maps, batch_gt, batch_metas)

    def predict(self, inputs, data_samples):
        feats = self.extract_feat(inputs)
        pred_maps = self.bbox_head(feats)
        batch_metas = [s.metainfo for s in data_samples] if data_samples else \
                      [dict(batch_input_shape=tuple(inputs.shape[-2:]))] * inputs.shape[0]
        return self.bbox_head.predict_by_feat(pred_maps, batch_metas)
```

### 3.7 data_samples schema(本轮 mock data 契约)

每个 `data_sample` 是 `mmengine.structures.BaseDataElement`:
- `gt_instances: InstanceData`,字段:
  - `bboxes: Tensor[Ni, 4]`,xyxy 像素单位
  - `labels: Tensor[Ni]`,int64,值域 `[0, num_classes-1]`
- `metainfo: dict`,至少含 `batch_input_shape: Tuple[int, int]`,可选 `img_shape / scale_factor`

## 4. 数据流

### 4.1 训练路径(`mode='loss'`)

```
inputs [B, 3, H, W] + data_samples (List[BaseDataElement])
  │
  └─→ SingleStageDetector.loss(inputs, data_samples)
        │
        ├─ feats = extract_feat(inputs)           # (P3, P4, P5)
        ├─ pred_maps = bbox_head(feats)           # 三层 [B, 3*(nc+5), Hi, Wi]
        ├─ batch_gt = [s.gt_instances for s in data_samples]
        ├─ batch_metas = [s.metainfo for s in data_samples]
        └─→ head.loss_by_feat(pred_maps, batch_gt, batch_metas)
              │
              ├─ featmap_sizes = [(Hi, Wi) for each layer]
              ├─ anchors_per_layer = prior_generator.grid_priors(featmap_sizes)
              │
              ├─ assignments = assigner(batch_gt, anchors_per_layer, featmap_sizes)
              │
              ├─ FOR each layer i:
              │     raw = pred_maps[i].view(B, na, nc+5, Hi, Wi).permute(0,1,3,4,2)
              │
              │     # 从 assignments[i] 取字段
              │     a = assignments[i]                          # 见 §3.3 schema
              │     matched_anchor = anchors_per_layer[i][a['anchor_idx']]   # (M, 2)
              │
              │     # 正样本位置 pred
              │     pos     = raw[a['img_idx'], a['anchor_idx'],
              │                   a['grid_y'], a['grid_x']]    # (M, nc+5)
              │     pos_xy  = pos[:, 0:2].sigmoid() * 2 - 0.5
              │     pos_wh  = (pos[:, 2:4].sigmoid() * 2) ** 2 * matched_anchor
              │     pos_obj = pos[:, 4]
              │     pos_cls = pos[:, 5:]
              │
              │     # bbox loss: 在网格单位下与 (gt_xy - grid_xy_int, gt_wh) 算 CIoU
              │     grid_xy_int = stack([a['grid_x'], a['grid_y']], -1).float()
              │     pred_box    = cat([pos_xy, pos_wh], -1)
              │     target_box  = cat([a['gt_xy'] - grid_xy_int, a['gt_wh']], -1)
              │     ciou        = bbox_ciou(cxcywh_to_xyxy(pred_box),
              │                             cxcywh_to_xyxy(target_box))
              │     loss_bbox  += (1.0 - ciou).mean()
              │
              │     # obj target: 正样本位置 = ciou.detach().clamp(0),其余 = 0
              │     obj_target = zeros(B, na, Hi, Wi)
              │     obj_target[a['img_idx'], a['anchor_idx'],
              │                a['grid_y'], a['grid_x']] = ciou.detach().clamp(0)
              │     loss_obj_layer = BCE_with_logits(raw[..., 4], obj_target).mean()
              │     loss_obj      += loss_obj_layer * obj_level_weights[i]
              │
              │     # cls loss: 仅 num_classes > 1 且 M > 0 时计算
              │     IF num_classes > 1 AND M > 0:
              │         cls_target = one_hot(a['gt_class'], nc).float()
              │         loss_cls  += BCE_with_logits(pos_cls, cls_target).mean()
              │
              ├─ loss_bbox *= loss_box_weight * B
              ├─ loss_obj  *= loss_obj_weight * B
              ├─ loss_cls  *= loss_cls_weight * B
              └─ return dict(loss_bbox, loss_obj, loss_cls)
```

batch_size 缩放(`* B`)是 ultralytics 的 convention,与默认 lr 配套使用。

### 4.2 推理路径(`mode='predict'`)

```
inputs [B, 3, H, W] + data_samples
  │
  └─→ SingleStageDetector.predict(inputs, data_samples)
        │
        ├─ feats = extract_feat(inputs)
        ├─ pred_maps = bbox_head(feats)
        ├─ batch_metas = [s.metainfo for s in data_samples]
        └─→ head.predict_by_feat(pred_maps, batch_metas)
              │
              ├─ featmap_sizes = [(Hi, Wi) for each layer]
              ├─ anchors = prior_generator.grid_priors(featmap_sizes)
              ├─ grid_xy = prior_generator.grid_xy(featmap_sizes)
              │
              ├─ FOR each layer i:
              │     raw = pred_maps[i].view(B, na, nc+5, Hi, Wi).permute(0,1,3,4,2)
              │     xy    = (raw[..., 0:2].sigmoid() * 2 - 0.5 + grid_xy[i]) * stride[i]
              │     wh    = (raw[..., 2:4].sigmoid() * 2) ** 2 * anchors[i] * stride[i]
              │     obj   = raw[..., 4].sigmoid()
              │     cls   = raw[..., 5:].sigmoid()
              │     score = obj.unsqueeze(-1) * cls            # (B, na, Hi, Wi, nc)
              │     bbox  = cxcywh_to_xyxy(cat([xy, wh], -1))
              │
              │     # 各层 flatten 到 (B, na*Hi*Wi, ...) 并按 batch 维度收集
              │
              ├─ 拼接三层后,对每张图独立:
              │     1) labels = score.argmax(-1);  top_scores = score.max(-1)
              │     2) score > score_thr 过滤
              │     3) batched_nms(bbox, top_scores, labels, iou_thr=nms_iou_thr)
              │     4) topk to max_per_img
              │
              └─ return [InstanceData(bboxes, scores, labels) for each img]
```

### 4.3 端到端串联示例

```python
from mmengine.structures import BaseDataElement, InstanceData

inputs = torch.randn(2, 3, 640, 640)
data_samples = [
    BaseDataElement(
        gt_instances=InstanceData(
            bboxes=torch.tensor([[10., 20., 100., 200.], [50., 50., 150., 250.]]),
            labels=torch.tensor([0, 1]),
        ),
        metainfo=dict(batch_input_shape=(640, 640), img_shape=(640, 640)),
    ),
    BaseDataElement(
        gt_instances=InstanceData(
            bboxes=torch.tensor([[200., 300., 400., 500.]]),
            labels=torch.tensor([2]),
        ),
        metainfo=dict(batch_input_shape=(640, 640), img_shape=(640, 640)),
    ),
]

model = MODELS.build(yolov5_detector_cfg)
losses = model.forward(inputs, data_samples, mode='loss')
loss   = sum(losses.values())
loss.backward()                                          # 不报错

preds  = model.forward(inputs, data_samples, mode='predict')
# preds = [InstanceData(bboxes=..., scores=..., labels=...), ...] 长度 = 2
```

## 5. 错误处理

延续上一轮原则:**只在构造时做防御性检查,运行时让 PyTorch / 业务异常直接抛。**

### 5.1 构造时校验

**YOLOv5AnchorGenerator:**
- `len(base_sizes) == len(strides)`,否则 AssertionError
- 每个 `base_sizes[i]` 长度一致,否则 AssertionError
- `strides` 所有值 > 0,否则 ValueError

**YOLOv5BBoxCoder:** 无构造参数,无校验。

**YOLOv5BatchAssigner:**
- `num_classes >= 1`,否则 ValueError
- `num_base_priors >= 1`,否则 ValueError
- `prior_match_thr > 0`,否则 ValueError
- `0 < near_neighbor_thr < 1`,否则 ValueError

**YOLOv5Head(新增参数):**
- `len(strides) == 3`,否则 AssertionError
- `len(obj_level_weights) == 3`,否则 AssertionError
- `0 < score_thr < 1`,否则 ValueError
- `0 < nms_iou_thr < 1`,否则 ValueError
- `max_per_img >= 1`,否则 ValueError
- `loss_box/obj/cls_weight >= 0`,否则 ValueError

### 5.2 运行时退化情况

**loss_by_feat 中"无任何正样本"**(batch 里所有 GT 都没被匹配上):
- `loss_bbox = 0.0`(tensor,与 pred 同 device/dtype,保持在计算图里)
- `loss_obj` 正常算(全位置 target=0,只算负样本)
- `loss_cls = 0.0`
- 不抛异常,不打 warning

**predict_by_feat 中"无任何检测过 score_thr"**:
- 该图返回 `InstanceData(bboxes=zeros(0,4), scores=zeros(0), labels=zeros(0, dtype=int64))`
- 不抛异常

### 5.3 不做的事

- 不做形状 / 类型校验(`bboxes` 是否真 xyxy、`labels` 是否 int64)— 上游契约
- 不做 `inputs.shape[-2:]` 与 `batch_input_shape` 一致性校验
- 不做 anchor 与 stride 自洽性校验
- 不做 NaN / Inf 检测

### 5.4 唯一例外:mock data 定位用 assert

`loss_by_feat` 入口加一句 `assert len(batch_gt_instances) == pred_maps[0].shape[0]`,错误消息含两个具体数字。便于 mock data 写错时快速定位。仅此一处。

## 6. 测试

目标:**单元逻辑正确 + 端到端跑通**,不做与 ultralytics 数值对齐(等下一轮 .pt 权重转换时做端到端对齐)。

### 6.1 测试文件

```
tests/
├── test_models.py              # 已有,21 个 test 保留;改写 TestDetector 2 个 raise 测试
└── test_yolov5_train.py        # 新增,本轮全部新测试都进此文件
```

### 6.2 新增测试用例清单(18 个)

**TestAnchorGenerator(3 个):**

1. `test_grid_priors_shape` — 三层 base_sizes / strides → 输出 3 个 Tensor `(3, ny, nx, 2)`,数值 = `base_sizes[i] / strides[i]`(手算对照)
2. `test_grid_xy_shape_and_values` — 输出三层 `(ny, nx, 2)`,验证 `grid_xy[0,0,0] == (0, 0)`,`grid_xy[0, ny-1, nx-1] == (nx-1, ny-1)`
3. `test_invalid_strides_raises` — strides 含 0 → ValueError

**TestBBoxCoder(3 个):**

4. `test_decode_known_values` — pred=zeros / anchor=(4,4) / grid=(5,5) / stride=8 → 中心 `(5+0.5)*8=44`,wh `(2*0.5)^2*4*8=8`
5. `test_encode_decode_roundtrip` — 随机 gt_xywh → encode → decode 还原(误差 < 1e-4)
6. `test_decode_batched_shape` — pred shape `(2, 3, 20, 20, 4)` → decode 输出最后一维 4

**TestAssigner(3 个):**

7. `test_assigner_basic_match` — 1 张图 1 个 GT,GT wh 与某层 anchor 比 1:1 → 该层 M > 0
8. `test_assigner_3grid_expansion` — GT 中心在格子内部 → 最多扩展 3 个 grid cell
9. `test_assigner_empty_batch_gt` — 整 batch 无 GT → 三层都返回 dict,各字段都是空 tensor(`numel() == 0`),不抛异常

**TestCIoU(2 个):**

10. `test_ciou_identical_boxes` — pred == target → ciou == 1.0
11. `test_ciou_disjoint_boxes` — 两个不相交的 box → ciou < 0(distance penalty)

**TestHeadLossByFeat(3 个):**

12. `test_loss_by_feat_basic` — 真实 head + 2 张图 + 几个 GT → 返回 dict 三个 loss,每个有限正数(或 0),能 `.backward()`
13. `test_loss_by_feat_all_empty_gt` — 所有图都无 GT → `loss_bbox=0, loss_cls=0, loss_obj > 0`,能 `.backward()`
14. `test_loss_by_feat_batch_mismatch_raises` — `len(batch_gt_instances) != B` → AssertionError

**TestHeadPredictByFeat(1 个):**

15. `test_predict_by_feat_returns_instancedata` — 真实 head + random pred_maps → 返回 List[InstanceData] 长度 = B,每个含 `bboxes/scores/labels`,shape / dtype 正确

**TestDetectorEndToEnd(3 个):**

16. `test_detector_loss_mode_runs` — `mode='loss'` 返回 dict 三 loss,sum 后 `.backward()`,backbone 参数收到非零梯度
17. `test_detector_predict_mode_returns_instancedata` — `mode='predict'` 返回 List[InstanceData] 长度 = B
18. `test_detector_unknown_mode_raises` — `mode='wrong'` → ValueError(基类 §3.6 行为)

### 6.3 旧测试改动

`tests/test_models.py::TestDetector` 中:
- `test_detector_loss_mode_raises` — 删除(现在 mode='loss' 不再 raise,新行为由 test 16 覆盖)
- `test_detector_predict_mode_raises` — 删除(同上,新行为由 test 17 覆盖)
- `test_detector_tensor_mode_end_to_end` — 保留不动

### 6.4 不测的东西

- 不与 ultralytics 数值对照(下一轮 .pt 权重转换时做端到端对齐)
- 不测 autoanchor(范围外)
- 不测 EMA / Mosaic / 数据增强(范围外)
- 不测训练实际收敛(范围外,需 dataset pipeline)
- 不测梯度数值正确性(只测有梯度,不测梯度值对错)

### 6.5 验证

```bash
pytest tests/test_yolov5_train.py -v
pytest tests/ -v
```

`tests/` 总数:34(已有)+ 18(新)- 2(删)= **50 passed** 即本轮交付完成。
