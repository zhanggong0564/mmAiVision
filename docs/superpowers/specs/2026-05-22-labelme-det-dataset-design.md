# LabelmeDetDataset 设计文档

- **日期**: 2026-05-22
- **作者**: zhanggong
- **状态**: Draft → 待用户确认
- **关联任务**: 实现 X-AnyLabeling/Labelme JSON 格式的目标检测数据加载；顺便把工程 scope 从 `mmengine_template` 重命名为 `mmaivision`

## 1. 背景与目标

工程目前是基于 [MMEngine Template](https://github.com/open-mmlab/mmengine-template) 拷贝来的脚手架。`mmengine_template/datasets/datasets.py` 中的 `YoLoDataset` 仅有空壳（`load_data_list` 未实现且名称与实际格式不符）。

需求：给定 [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling) / Labelme 风格的 JSON 标注，每张图一个 JSON，写一个可被 MMEngine Runner 直接 build 的目标检测数据集，输出字段对齐 mmdet 习惯，便于复用 mmdet 的 transforms / `PackDetInputs`。

参考示例 JSON（来自 `IMG_5266.json`）：

```json
{
  "version": "3.2.1",
  "shapes": [
    {
      "label": "dc_line",
      "points": [[x1,y1], [x2,y1], [x2,y2], [x1,y2]],
      "shape_type": "rectangle",
      "difficult": false,
      ...
    }, ...
  ],
  "imagePath": "..\\IMG_5266.jpg",
  "imageHeight": 3024,
  "imageWidth": 4032
}
```

非目标：
- 不实现 transforms / 模型 / 训练流程；只做数据加载
- 不引入 mmdet 硬依赖
- 不动 `transforms.py` 既有空壳
- 不实现 KIE / group_id / attributes 等高级字段

## 2. 整体方案

**方案 A — 纯 MMEngine BaseDataset**（用户已确认）

- 新建 `LabelmeDetDataset` 继承 `mmengine.dataset.BaseDataset`，通过 `@DATASETS.register_module()` 注册
- 解析逻辑封装为类的私有方法 `_parse_one(json_path) -> dict`
- 字段命名照 mmdet 走，未来直接套 `mmdet.PackDetInputs` 不用改 dataset

理由：当前工程规模小，不引 mmdet 依赖最干净；解析逻辑作为类私有方法集中阅读；mmdet 兼容字段名零迁移成本。

## 3. 目录与路径约定

构造参数（透传给 `BaseDataset`）:

```python
LabelmeDetDataset(
    ann_file='train.txt',                              # 相对 data_root
    data_root='data/my_dataset',
    data_prefix=dict(img='images', ann='annotations'), # 与 mmdet 一致：dict 形式
    metainfo=dict(classes=('dc_line', ...)),
    filter_cfg=dict(filter_empty_gt=True, min_size=1),
    pipeline=[...],
    test_mode=False,
    serialize_data=True,
    lazy_init=False,
)
```

磁盘布局假设：

```
data_root/
├── images/
│   ├── IMG_5266.jpg
│   └── IMG_5267.jpg
├── annotations/
│   ├── IMG_5266.json
│   └── IMG_5267.json
└── train.txt        # 每行一个 stem（如 "IMG_5266"）或相对 annotations 的 json 相对路径
```

`ann_file` 行的解析规则：
- 若以 `.json` 结尾：视作相对 `{data_root}/{data_prefix['ann']}` 的相对路径
- 否则：视作 stem，json 路径 = `{data_root}/{data_prefix['ann']}/{stem}.json`
- 行首 `#` 视作注释，空行跳过

图片路径解析：
- 不直接拼接 JSON 内的 `imagePath`（其 `..\\` 是 Windows 反斜杠且与子目录布局不匹配）
- 用 `Path(imagePath).name` 取 basename（仅为了拿扩展名）
- 最终图片路径 = `{data_root}/{data_prefix['img']}/{basename}`
- 若 `imagePath` 字段缺失：fallback 用 `{stem}.jpg`

## 4. data_info 字典 Schema

`load_data_list()` 返回 `List[dict]`，每元素：

```python
{
    'img_path': str,        # 绝对路径
    'img_id': str,          # stem（IMG_5266）
    'height': int,          # 来自 JSON.imageHeight
    'width': int,           # 来自 JSON.imageWidth
    'instances': [
        {
            'bbox': [x1, y1, x2, y2],     # float, xyxy, 像素坐标
            'bbox_label': int,             # classes.index(label)
            'ignore_flag': int,            # 1 if difficult else 0
            'mask': [[x1,y1,x2,y2,...]],   # 仅 polygon 时存在；外层 list 是 COCO 多 polygon 格式
        }, ...
    ],
}
```

字段计算规则：

| 输出字段 | 计算方式 |
|----------|---------|
| `bbox` | 一律 `[min(xs), min(ys), max(xs), max(ys)]`，rectangle/polygon 通用；不假定点序 |
| `bbox_label` | `classes.index(shape['label'])`；不在 classes 中 → 跳过该 shape + 累计 warn |
| `ignore_flag` | `int(shape.get('difficult', False))` |
| `mask` | shape_type=='polygon' 时把 points 展平为 `[x1,y1,x2,y2,...]` 一维 list 并外包一层 list；rectangle 时不写此字段 |

shape 过滤规则：
- `shape_type` 不是 `rectangle` 或 `polygon` → 跳过 + 累计 warn
- bbox 宽或高 ≤ 0 → 跳过

## 5. 配置接口

### 5.1 `metainfo`

- 类属性 `METAINFO = dict(classes=None)`
- `classes` 必须由 config 提供：构造时若 `self._metainfo['classes']` 为 None → `raise ValueError('classes must be specified via metainfo')`

### 5.2 `filter_cfg`

重写 `filter_data()` 实现：

- `filter_empty_gt: bool = True`：训练模式下（`test_mode=False`）丢弃 0 个有效 instance 的样本；test 模式始终保留全部，便于评测
- `min_size: int = 1`：加载阶段过滤 `bbox_w < min_size or bbox_h < min_size` 的 instance；默认 1 等价于不过滤

### 5.3 异常与日志

- 单个 JSON 读取失败（文件不存在 / JSON 解析失败）→ `MMLogger.get_current_instance().warning(f'skip {path}: {e}')`，该样本整体跳过，不抛
- 未知 label / 非法 shape_type / 非法 bbox → 全部加载完后汇总打一条 warning（避免逐条刷屏），格式：`f'skipped {n_unknown} shapes with unknown labels, {n_bad_type} with unsupported shape_type, {n_bad_bbox} with invalid bbox'`
- 加载结束打一条 info：`f'loaded {n_samples} samples, {n_instances} instances; per-class counts: {...}'`
- 不在 dataset 层校验图片可读，下放给 pipeline 的 `LoadImageFromFile`

## 6. 文件改动清单

### 6.1 主任务：数据集实现

1. **重写** `mmengine_template/datasets/datasets.py`
   - 移除 `YoLoDataset` 占位
   - 实现 `LabelmeDetDataset(BaseDataset)`：`load_data_list()`、`_parse_one()`、`filter_data()`、`METAINFO`

2. **更新** `mmengine_template/datasets/__init__.py`
   - `from .datasets import LabelmeDetDataset`
   - 更新 `__all__`（移除 `CustomDataset`，加入 `LabelmeDetDataset`；保留 `CustomTransform`）

3. **写入** `configs/_base_/dataset.py`（当前空文件）
   - 最小可用的 dataset config 示例：
     - `dataset_type = 'LabelmeDetDataset'`、`data_root`、`metainfo=dict(classes=(...))`
     - train/val/test dataloader 三件套
     - pipeline 极简：`[dict(type='LoadImageFromFile')]`（避免引入 mmdet 依赖；用户自行追加 transforms）

4. **新建测试 fixture** `tests/data/labelme_sample/`
   - `images/IMG_a.jpg`、`images/IMG_b.jpg`（任意 PNG header 占位文件即可，dataset 不读图）
   - `annotations/IMG_a.json`：含 2 个 rectangle（一个 difficult）、1 个 polygon、1 个未知 label、1 个非法 shape_type
   - `annotations/IMG_b.json`：含 0 个有效 instance（用于 filter_empty_gt 测试）
   - `annotations/IMG_corrupt.json`：故意写成非法 JSON
   - `train.txt`：列出 IMG_a / IMG_b / IMG_corrupt 三个 stem

5. **新建** `tools/verify_dataset.py` — 验证 demo 脚本（参考 mmdet 的 `tools/misc/browse_dataset.py` 思路，但更轻）
   - 用法：`python tools/verify_dataset.py <config> [--out-dir vis] [--num 5] [--split train]`
   - 流程：
     1. `Config.fromfile(config)` → 取 `cfg.train_dataloader.dataset`（`--split val` 时取 val）
     2. `DATASETS.build(dataset_cfg)` 构建实例
     3. 打印：`len(ds)`、`ds.metainfo['classes']`、按类 instance 计数（基于 `ds.data_list`，**不走 pipeline**）
     4. 随机/前 `--num` 个样本：
        - 打印 `data_info` 字典关键字段（img_path、height、width、instances 数量、前 3 个 bbox）
        - 用 `cv2.imread(img_path)` 读图，画每个 instance：
          - rectangle bbox 用绿色框
          - polygon 额外用蓝色描边 `cv2.polylines`
          - `ignore_flag=1` 的用灰色框 + "(diff)" 后缀
          - label text 用 `cv2.putText` 标在 bbox 左上
        - 保存到 `{out_dir}/{img_id}.jpg`
   - 仅依赖 `opencv-python`（项目大概率已装；未装则 `ImportError` 时给出友好提示）
   - **不走 pipeline**：直接读 `ds.data_list[i]` 拿原始 dict 渲染，避免被 transforms 缩放/翻转干扰，目的是验证 loader 本身

6. **新建测试** `tests/test_datasets.py`，含以下用例：
   - `test_load_data_list_basic` — 样本数、img_path 拼接正确、height/width 从 JSON 正确读取
   - `test_bbox_from_rectangle_points_unordered` — 故意打乱 4 点顺序，bbox 仍是 [min_x, min_y, max_x, max_y]
   - `test_polygon_creates_mask_and_bbox` — polygon 同时产出 bbox + mask 字段；rectangle 不产出 mask 字段
   - `test_difficult_sets_ignore_flag` — `difficult=True` → `ignore_flag=1`
   - `test_unknown_label_skipped` — 未在 classes 的 shape 被跳过，warning 被触发
   - `test_invalid_shape_type_skipped` — `shape_type` 既不是 rectangle 也不是 polygon 的 shape 被跳过
   - `test_min_size_filter` — 小于 `min_size` 的 instance 被丢
   - `test_filter_empty_gt_train_mode` — train 模式 0 instance 样本被过滤；test 模式保留
   - `test_missing_classes_raises` — 不传 `metainfo.classes` 抛 `ValueError`
   - `test_corrupt_json_skipped_not_raised` — 损坏 JSON 不抛，仅 warn

不改动：`transforms.py`、`registry.py`、`tools/`、其他子包。

### 6.2 子任务：scope 重命名 `mmengine_template` → `mmaivision`

机械重命名，分两步：

**Step A：包目录与 demo 文件改名**
- `mv mmengine_template/ mmaivision/`
- `mv demo/mmengine_template_demo.py demo/mmaivision_demo.py`

**Step B：字符串替换**（在以下 21 个文件中把 `mmengine_template` 替换为 `mmaivision`；另检查 `MANIFEST.in`、`setup.py`、`setup.cfg`、`configs/_base_/default_runtime.py`）：

```
MANIFEST.in
demo/mmengine_template_demo.py  → demo/mmaivision_demo.py
mmengine_template/registry.py
mmengine_template/datasets/datasets.py
mmengine_template/datasets/transforms.py
mmengine_template/engine/hooks.py
mmengine_template/engine/optim_wrapper_constructors.py
mmengine_template/engine/optim_wrappers.py
mmengine_template/engine/optimizers.py
mmengine_template/engine/schedulers.py
mmengine_template/evaluation/evaluator.py
mmengine_template/evaluation/metrics.py
mmengine_template/infer/inference.py
mmengine_template/models/model.py
mmengine_template/models/weight_init.py
mmengine_template/models/wrappers.py
configs/_base_/default_runtime.py
tools/test.py
setup.py
setup.cfg
```

替换后 sanity check：
- `grep -r mmengine_template .` 应只在 docs/spec 中残留
- `python -c "from mmaivision.registry import DATASETS; print(DATASETS)"` 不报错
- `pip install -e .` 可重新安装

`__init__.py` 中 `CustomDataset` → `LabelmeDetDataset` 的改动在主任务 step 2 完成（避免双重改动）。

## 7. 实现顺序建议

1. **先做 scope 重命名**（§6.2）— 机械工作但影响所有 import，单独成步避免和功能改动混在一起难 review
2. **再实现数据集**（§6.1 step 1-3）
3. **写测试 fixture 和单元测试**（§6.1 step 4 + step 6）
4. **写验证 demo**（§6.1 step 5）— 用真实数据跑一遍，保存可视化图片肉眼看 bbox/polygon 是否对齐
5. **手工 smoke test**：用 `Config.fromfile('configs/_base_/dataset.py')` + `DATASETS.build(cfg.train_dataloader.dataset)` 验证可以 build，`len(ds)` 和 `ds[0]` 合理

## 8. 依赖

- 数据集本身：只用 stdlib（`json`、`pathlib`、`logging`），不新增外部依赖
- `tools/verify_dataset.py` 需要 `opencv-python`（项目通常已装；未装则 ImportError 时提示用户安装）

## 9. 风险与备选

- **风险 1：用户实际数据布局不是 images/ + annotations/ 而是混合目录** — 已在 §3 的 `data_prefix` 设计中预留可配置；若两者同目录，用户只需 `data_prefix=dict(img='', ann='')`
- **风险 2：JSON imagePath 实际指向了正确的相对路径** — 当前设计选择不信任 imagePath 的目录部分（只取 basename）；若未来需要支持，可加配置 `trust_image_path: bool`
- **风险 3：polygon 输出格式与未来要用的 mmdet transform 不匹配** — mmdet 的 `LoadAnnotations(with_mask=True)` 期待 instances[i]['mask'] 为 `List[List[float]]`，已对齐
