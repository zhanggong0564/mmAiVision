# LabelmeDetDataset 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现一个加载 X-AnyLabeling / Labelme 风格 JSON 标注的目标检测 Dataset（`LabelmeDetDataset`），输出对齐 mmdet 字段；顺带把工程 scope 从 `mmengine_template` 重命名为 `mmaivision`，并提供单元测试和一个可视化 verify demo。

**Architecture:** 继承 `mmengine.dataset.BaseDataset`，注册到 `DATASETS`。`load_data_list()` 读取 txt 索引 → 逐个调 `_parse_one(json_path)` 解析 labelme JSON 为 mmdet 风格 dict（`img_path/img_id/height/width/instances`）；`filter_data()` 重写以支持 `filter_empty_gt`。验证脚本 `tools/verify_dataset.py` 直接读 `ds.data_list` 渲染 bbox/polygon 到图片。

**Tech Stack:** Python 3.x, MMEngine BaseDataset, pytest (单测), OpenCV (仅 verify demo 用)。无新增运行时依赖。

**关键文件路径（重命名后）：**
- `mmaivision/datasets/datasets.py` — `LabelmeDetDataset` 实现
- `mmaivision/datasets/__init__.py` — 导出
- `configs/_base_/dataset.py` — 配置示例
- `tools/verify_dataset.py` — 验证 demo
- `tests/test_datasets.py` — 单元测试
- `tests/data/labelme_sample/` — 测试 fixture

---

## 前置约定

- 所有路径以仓库根 `/home/zhanggong/workspace/AIVision/mmAiVsionT/` 为基准
- 每个 Task 末尾的 commit 都用**中文** message，**不带 Co-Authored-By trailer**
- 测试用例文件 `tests/test_datasets.py` 会随任务 4-12 逐步追加；任务 13 之前不要 `pytest tests/`（运行单条用例就好，避免触发未完成代码）

---

## Task 1: 重命名 scope `mmengine_template` → `mmaivision`

**Files:**
- Rename: 目录 `mmengine_template/` → `mmaivision/`
- Rename: `demo/mmengine_template_demo.py` → `demo/mmaivision_demo.py`
- Modify (字符串替换): `setup.py`、`setup.cfg`、`MANIFEST.in`、`configs/_base_/default_runtime.py`、`tools/test.py`，以及移动后 `mmaivision/` 树下所有 `.py`

**Test:** sanity check（不是 pytest，是 grep + import 验证）

- [ ] **Step 1: 重命名包目录和 demo 文件**

```bash
cd /home/zhanggong/workspace/AIVision/mmAiVsionT
git mv mmengine_template mmaivision
git mv demo/mmengine_template_demo.py demo/mmaivision_demo.py
```

- [ ] **Step 2: 批量替换文件内容中的 scope 字符串**

```bash
cd /home/zhanggong/workspace/AIVision/mmAiVsionT
find . -path ./.git -prune -o -type f \( -name "*.py" -o -name "*.cfg" -o -name "*.in" \) -print | xargs grep -l "mmengine_template" 2>/dev/null | xargs sed -i 's/mmengine_template/mmaivision/g'
```

注意：`find` 已通过 `-prune` 排除 `.git`；仅处理 `.py / .cfg / .in`，不动 `.md`（用户文档自己维护）。

- [ ] **Step 3: 验证替换完整**

Run:
```bash
grep -rl "mmengine_template" . --include="*.py" --include="*.cfg" --include="*.in" 2>/dev/null
```

Expected: 输出为空（无任何文件残留）。

- [ ] **Step 4: 验证新 scope 可 import**

Run:
```bash
cd /home/zhanggong/workspace/AIVision/mmAiVsionT && pip install -e . --quiet && python -c "from mmaivision.registry import DATASETS; print('OK', DATASETS)"
```

Expected: 打印 `OK Registry of dataset (...)` 之类，不抛 `ModuleNotFoundError`。

- [ ] **Step 5: 提交**

```bash
git add -A
git commit -m "refactor: 包名 mmengine_template 重命名为 mmaivision

机械重命名：目录、demo 文件名、所有 .py/.cfg/.in 文件内的
mmengine_template 字符串。setup.py 和 configs/_base_/default_runtime.py
中的 scope 字符串同步更新。"
```

---

## Task 2: 创建测试 fixture 文件

**Files:**
- Create: `tests/data/labelme_sample/train.txt`
- Create: `tests/data/labelme_sample/images/IMG_a.jpg` (占位文件)
- Create: `tests/data/labelme_sample/images/IMG_b.jpg` (占位文件)
- Create: `tests/data/labelme_sample/annotations/IMG_a.json`
- Create: `tests/data/labelme_sample/annotations/IMG_b.json`
- Create: `tests/data/labelme_sample/annotations/IMG_corrupt.json`
- Create: `tests/__init__.py` (空文件，方便 pytest 收集)

**Test:** 无（纯准备）

- [ ] **Step 1: 创建目录树**

```bash
mkdir -p tests/data/labelme_sample/images tests/data/labelme_sample/annotations
touch tests/__init__.py
```

- [ ] **Step 2: 写占位图片**

dataset 不读图，最小内容即可。用 1×1 PNG header 字节占位（也叫 JPG 就行）：

```bash
printf '\x89PNG\r\n\x1a\n' > tests/data/labelme_sample/images/IMG_a.jpg
printf '\x89PNG\r\n\x1a\n' > tests/data/labelme_sample/images/IMG_b.jpg
```

- [ ] **Step 3: 写 `train.txt`**

文件内容（每行一个 stem，含一个被注释行）：

```
# 注释行应被跳过
IMG_a
IMG_b
IMG_corrupt
```

写入：

```bash
cat > tests/data/labelme_sample/train.txt <<'EOF'
# 注释行应被跳过
IMG_a
IMG_b
IMG_corrupt
EOF
```

- [ ] **Step 4: 写 `IMG_a.json`**

内含：2 个 rectangle（其中第二个 `difficult=True`）、1 个 polygon、1 个未知 label、1 个非法 shape_type、1 个 zero-size rectangle（用于 min_size 测试）。

```bash
cat > tests/data/labelme_sample/annotations/IMG_a.json <<'EOF'
{
  "version": "3.2.1",
  "flags": {},
  "shapes": [
    {
      "label": "dc_line",
      "points": [[30, 40], [10, 40], [10, 20], [30, 20]],
      "shape_type": "rectangle",
      "difficult": false,
      "group_id": null,
      "description": "",
      "flags": {},
      "attributes": {},
      "kie_linking": []
    },
    {
      "label": "dc_line",
      "points": [[50, 60], [70, 60], [70, 80], [50, 80]],
      "shape_type": "rectangle",
      "difficult": true,
      "group_id": null,
      "description": "",
      "flags": {}
    },
    {
      "label": "fuse",
      "points": [[100, 100], [120, 100], [130, 110], [110, 120], [90, 110]],
      "shape_type": "polygon",
      "difficult": false,
      "group_id": null,
      "description": "",
      "flags": {}
    },
    {
      "label": "unknown_thing",
      "points": [[200, 200], [220, 200], [220, 220], [200, 220]],
      "shape_type": "rectangle",
      "difficult": false
    },
    {
      "label": "dc_line",
      "points": [[300, 300], [310, 310]],
      "shape_type": "circle",
      "difficult": false
    },
    {
      "label": "dc_line",
      "points": [[400, 400], [400, 400], [400, 400], [400, 400]],
      "shape_type": "rectangle",
      "difficult": false
    }
  ],
  "imagePath": "..\\IMG_a.jpg",
  "imageData": null,
  "imageHeight": 1000,
  "imageWidth": 1500
}
EOF
```

- [ ] **Step 5: 写 `IMG_b.json` (空 shapes，用于 filter_empty_gt 测试)**

```bash
cat > tests/data/labelme_sample/annotations/IMG_b.json <<'EOF'
{
  "version": "3.2.1",
  "flags": {},
  "shapes": [],
  "imagePath": "..\\IMG_b.jpg",
  "imageData": null,
  "imageHeight": 800,
  "imageWidth": 1200
}
EOF
```

- [ ] **Step 6: 写 `IMG_corrupt.json` (非法 JSON)**

```bash
cat > tests/data/labelme_sample/annotations/IMG_corrupt.json <<'EOF'
{this is not valid json
EOF
```

- [ ] **Step 7: 验证 fixture 内容存在**

Run:
```bash
ls tests/data/labelme_sample/annotations/ tests/data/labelme_sample/images/ && cat tests/data/labelme_sample/train.txt
```

Expected:
```
images/:
IMG_a.jpg
IMG_b.jpg

annotations/:
IMG_a.json
IMG_b.json
IMG_corrupt.json

# 注释行应被跳过
IMG_a
IMG_b
IMG_corrupt
```

- [ ] **Step 8: 提交**

```bash
git add tests/
git commit -m "test: 添加 LabelmeDetDataset 单元测试 fixture

包含 train.txt、占位图片以及三种 JSON 标注样本：正常标注、空 shapes、
损坏 JSON，覆盖 rectangle/polygon/difficult/未知 label/非法 shape_type
等场景。"
```

---

## Task 3: Scaffold `LabelmeDetDataset` — 类骨架 + classes 校验

**Files:**
- Modify: `mmaivision/datasets/datasets.py` (整体重写)
- Test: `tests/test_datasets.py` (新建)

- [ ] **Step 1: 写失败测试 `test_missing_classes_raises`**

创建 `tests/test_datasets.py`：

```python
"""LabelmeDetDataset 单元测试。"""
import os.path as osp

import pytest

import mmaivision  # noqa: F401 触发注册
from mmaivision.datasets.datasets import LabelmeDetDataset

FIXTURE = osp.join(osp.dirname(__file__), 'data', 'labelme_sample')


def _make_dataset(**overrides):
    """构造一个测试用的 LabelmeDetDataset。"""
    kwargs = dict(
        ann_file='train.txt',
        data_root=FIXTURE,
        data_prefix=dict(img='images', ann='annotations'),
        metainfo=dict(classes=('dc_line', 'fuse')),
        pipeline=[],
        lazy_init=False,
    )
    kwargs.update(overrides)
    return LabelmeDetDataset(**kwargs)


def test_missing_classes_raises():
    with pytest.raises(ValueError, match='classes'):
        LabelmeDetDataset(
            ann_file='train.txt',
            data_root=FIXTURE,
            data_prefix=dict(img='images', ann='annotations'),
            metainfo=dict(),  # 没有 classes
            pipeline=[],
            lazy_init=False,
        )
```

- [ ] **Step 2: 运行测试确认失败**

Run:
```bash
cd /home/zhanggong/workspace/AIVision/mmAiVsionT && pytest tests/test_datasets.py::test_missing_classes_raises -v
```

Expected: FAIL — 因为 `datasets.py` 里还没有 `LabelmeDetDataset` 类（当前是空壳 `YoLoDataset`）。

- [ ] **Step 3: 重写 `mmaivision/datasets/datasets.py` 加入类骨架**

完整替换为：

```python
"""LabelmeDetDataset: 加载 X-AnyLabeling / Labelme 风格 JSON 标注的目标检测数据集。"""
import json
import os.path as osp
from pathlib import Path
from typing import Any, Dict, List, Optional

from mmengine.dataset import BaseDataset
from mmengine.logging import MMLogger

from mmaivision.registry import DATASETS


@DATASETS.register_module()
class LabelmeDetDataset(BaseDataset):
    """Labelme/X-AnyLabeling 风格目标检测数据集。

    每个标注文件是一张图的 labelme JSON；`ann_file` 是 txt，每行一个 stem
    或相对 annotations 目录的 json 路径。输出字段对齐 mmdet 习惯，便于复用
    mmdet 的 transforms。
    """

    METAINFO: Dict[str, Any] = dict(classes=None)

    def load_data_list(self) -> List[Dict[str, Any]]:
        classes = self._metainfo.get('classes')
        if classes is None:
            raise ValueError(
                'classes must be specified via metainfo, e.g. '
                "metainfo=dict(classes=('dc_line', ...))")
        # 后续 task 实现
        return []
```

- [ ] **Step 4: 运行测试确认通过**

Run:
```bash
pytest tests/test_datasets.py::test_missing_classes_raises -v
```

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add mmaivision/datasets/datasets.py tests/test_datasets.py
git commit -m "feat(datasets): 新建 LabelmeDetDataset 骨架并校验 classes

继承 BaseDataset 并注册到 DATASETS。load_data_list 暂返回空列表，
仅实现 metainfo.classes 必填校验；缺失时抛 ValueError。"
```

---

## Task 4: 基础 rectangle 加载（img_path / height / width / bbox / bbox_label）

**Files:**
- Modify: `mmaivision/datasets/datasets.py`
- Test: `tests/test_datasets.py`

- [ ] **Step 1: 在 `tests/test_datasets.py` 末尾追加两个测试**

```python
def test_load_data_list_basic():
    ds = _make_dataset()
    # IMG_corrupt.json 解析失败被跳过 → 余下 IMG_a + IMG_b 共 2 个样本
    # IMG_b 暂时也保留（filter_empty_gt 由后续任务实现）
    assert len(ds.data_list) == 2

    a = next(d for d in ds.data_list if d['img_id'] == 'IMG_a')
    assert a['img_path'] == osp.join(FIXTURE, 'images', 'IMG_a.jpg')
    assert a['height'] == 1000
    assert a['width'] == 1500

    # rectangle bbox 来自 4 个角点 min/max
    rect_instances = [
        i for i in a['instances']
        if i['bbox_label'] == 0 and i['ignore_flag'] == 0
    ]
    assert len(rect_instances) >= 1
    bbox = rect_instances[0]['bbox']
    # IMG_a.json 的第一个 dc_line rect：(10, 20, 30, 40)
    assert bbox == [10.0, 20.0, 30.0, 40.0]


def test_bbox_from_rectangle_points_unordered():
    """fixture 中 IMG_a.json 第一个 rect 的 4 点是乱序的：
    [[30,40],[10,40],[10,20],[30,20]]，bbox 应是 [10,20,30,40]。"""
    ds = _make_dataset()
    a = next(d for d in ds.data_list if d['img_id'] == 'IMG_a')
    rect = next(
        i for i in a['instances']
        if i['bbox_label'] == 0 and i['ignore_flag'] == 0)
    x1, y1, x2, y2 = rect['bbox']
    assert x1 < x2 and y1 < y2
    assert (x1, y1, x2, y2) == (10.0, 20.0, 30.0, 40.0)
```

- [ ] **Step 2: 运行测试确认失败**

Run:
```bash
pytest tests/test_datasets.py::test_load_data_list_basic tests/test_datasets.py::test_bbox_from_rectangle_points_unordered -v
```

Expected: FAIL — `load_data_list` 返回空列表。

- [ ] **Step 3: 在 `datasets.py` 中实现 `load_data_list` 与 `_parse_one`**

替换 `load_data_list` 并新增 helper（追加到类内）：

```python
    def load_data_list(self) -> List[Dict[str, Any]]:
        classes = self._metainfo.get('classes')
        if classes is None:
            raise ValueError(
                'classes must be specified via metainfo, e.g. '
                "metainfo=dict(classes=('dc_line', ...))")

        logger = MMLogger.get_current_instance()
        data_list: List[Dict[str, Any]] = []
        ann_dir = self._resolve_prefix_dir('ann')
        img_dir = self._resolve_prefix_dir('img')

        for stem_or_path in self._iter_ann_file_lines():
            json_path = self._resolve_json_path(stem_or_path, ann_dir)
            stem = Path(json_path).stem
            try:
                data_info = self._parse_one(json_path, stem, img_dir, classes)
            except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
                logger.warning(f'skip {json_path}: {e}')
                continue
            data_list.append(data_info)

        logger.info(
            f'loaded {len(data_list)} samples from {self.ann_file}')
        return data_list

    def _resolve_prefix_dir(self, key: str) -> str:
        """从 self.data_prefix 取出目录，已是绝对路径则直接返回。

        BaseDataset._join_prefix 在某些版本里会把 data_prefix 的值与
        data_root 自动拼接成绝对路径，再次拼会变成双重 join。这里做
        防御性判断：仅当返回值还是相对路径时才拼 data_root。
        """
        raw = (self.data_prefix or {}).get(key, '')
        if not raw:
            return self.data_root
        if osp.isabs(raw):
            return raw
        return osp.join(self.data_root, raw)

    def _iter_ann_file_lines(self):
        with open(self.ann_file, 'r', encoding='utf-8') as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith('#'):
                    continue
                yield line

    @staticmethod
    def _resolve_json_path(stem_or_path: str, ann_dir: str) -> str:
        if stem_or_path.endswith('.json'):
            return osp.join(ann_dir, stem_or_path)
        return osp.join(ann_dir, f'{stem_or_path}.json')

    def _parse_one(self, json_path: str, stem: str, img_dir: str,
                   classes) -> Dict[str, Any]:
        with open(json_path, 'r', encoding='utf-8') as f:
            obj = json.load(f)

        image_path_field = obj.get('imagePath') or f'{stem}.jpg'
        img_basename = Path(image_path_field.replace('\\', '/')).name
        img_path = osp.join(img_dir, img_basename)

        instances: List[Dict[str, Any]] = []
        for shape in obj.get('shapes', []):
            inst = self._parse_shape(shape, classes)
            if inst is not None:
                instances.append(inst)

        return dict(
            img_path=img_path,
            img_id=stem,
            height=int(obj['imageHeight']),
            width=int(obj['imageWidth']),
            instances=instances,
        )

    def _parse_shape(self, shape: Dict[str, Any],
                     classes) -> Optional[Dict[str, Any]]:
        label = shape.get('label')
        if label not in classes:
            return None
        shape_type = shape.get('shape_type')
        if shape_type != 'rectangle':
            return None
        points = shape.get('points', [])
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
        if not xs or not ys:
            return None
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        if (x2 - x1) <= 0 or (y2 - y1) <= 0:
            return None
        return dict(
            bbox=[x1, y1, x2, y2],
            bbox_label=classes.index(label),
            ignore_flag=int(shape.get('difficult', False)),
        )
```

注意：`_parse_shape` 此时只处理 rectangle；polygon 在 Task 6 加入；unknown label / 非法 shape 的 warning 聚合在 Task 8-9 加入；min_size 在 Task 10 加入。

- [ ] **Step 4: 运行两个测试确认通过**

Run:
```bash
pytest tests/test_datasets.py::test_load_data_list_basic tests/test_datasets.py::test_bbox_from_rectangle_points_unordered -v
```

Expected: 2 passed。

注意：`test_load_data_list_basic` 断言 `len(ds.data_list) == 2`：IMG_corrupt 在解析阶段抛 `JSONDecodeError` 被跳过，剩下 IMG_a 和 IMG_b。

- [ ] **Step 5: 提交**

```bash
git add mmaivision/datasets/datasets.py tests/test_datasets.py
git commit -m "feat(datasets): 实现 rectangle 标注的基础解析

load_data_list 读取 txt 索引并逐个解析 JSON；从 4 角点 min/max 算 bbox，
不依赖点序。损坏 JSON 在此阶段已被静默跳过（异常吞并 warn）。"
```

---

## Task 5: `difficult` → `ignore_flag`

**Files:**
- Test: `tests/test_datasets.py`

(实现已在 Task 4 的 `_parse_shape` 中完成；本 Task 仅添加测试覆盖。)

- [ ] **Step 1: 追加测试 `test_difficult_sets_ignore_flag`**

```python
def test_difficult_sets_ignore_flag():
    ds = _make_dataset()
    a = next(d for d in ds.data_list if d['img_id'] == 'IMG_a')
    flags = sorted(
        i['ignore_flag'] for i in a['instances'] if i['bbox_label'] == 0)
    # IMG_a 有两个 dc_line rectangle：一个 difficult=False, 一个 difficult=True
    # (第三个角点重合的 rect 因 bbox 非法已被丢弃)
    assert flags == [0, 1]
```

- [ ] **Step 2: 运行测试确认通过**

Run:
```bash
pytest tests/test_datasets.py::test_difficult_sets_ignore_flag -v
```

Expected: PASS。

- [ ] **Step 3: 提交**

```bash
git add tests/test_datasets.py
git commit -m "test: 补充 difficult 字段映射 ignore_flag 的测试用例"
```

---

## Task 6: polygon 支持（生成 bbox + mask 字段）

**Files:**
- Modify: `mmaivision/datasets/datasets.py` (`_parse_shape`)
- Test: `tests/test_datasets.py`

- [ ] **Step 1: 追加测试 `test_polygon_creates_mask_and_bbox`**

```python
def test_polygon_creates_mask_and_bbox():
    ds = _make_dataset()
    a = next(d for d in ds.data_list if d['img_id'] == 'IMG_a')
    polys = [i for i in a['instances'] if i['bbox_label'] == 1]
    # fixture: 一个 fuse polygon
    assert len(polys) == 1
    p = polys[0]
    # bbox 来自所有点的 min/max；IMG_a fuse points:
    # [[100,100],[120,100],[130,110],[110,120],[90,110]]
    assert p['bbox'] == [90.0, 100.0, 130.0, 120.0]
    # mask: List[List[float]]，外层一段 polygon
    assert 'mask' in p
    assert isinstance(p['mask'], list) and len(p['mask']) == 1
    flat = p['mask'][0]
    assert flat == [100.0, 100.0, 120.0, 100.0, 130.0, 110.0,
                    110.0, 120.0, 90.0, 110.0]

    # rectangle 不应有 mask 字段
    rects = [
        i for i in a['instances']
        if i['bbox_label'] == 0 and i['ignore_flag'] == 0
    ]
    assert all('mask' not in r for r in rects)
```

- [ ] **Step 2: 运行测试确认失败**

Run:
```bash
pytest tests/test_datasets.py::test_polygon_creates_mask_and_bbox -v
```

Expected: FAIL — polygon 当前被 `_parse_shape` 当成非法 shape_type 返回 None。

- [ ] **Step 3: 修改 `_parse_shape` 支持 polygon**

把 `_parse_shape` 改为：

```python
    def _parse_shape(self, shape: Dict[str, Any],
                     classes) -> Optional[Dict[str, Any]]:
        label = shape.get('label')
        if label not in classes:
            return None
        shape_type = shape.get('shape_type')
        if shape_type not in ('rectangle', 'polygon'):
            return None
        points = shape.get('points', [])
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
        if not xs or not ys:
            return None
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        if (x2 - x1) <= 0 or (y2 - y1) <= 0:
            return None
        inst: Dict[str, Any] = dict(
            bbox=[x1, y1, x2, y2],
            bbox_label=classes.index(label),
            ignore_flag=int(shape.get('difficult', False)),
        )
        if shape_type == 'polygon':
            flat = [coord for xy in zip(xs, ys) for coord in xy]
            inst['mask'] = [flat]
        return inst
```

- [ ] **Step 4: 运行测试确认通过**

Run:
```bash
pytest tests/test_datasets.py::test_polygon_creates_mask_and_bbox -v
```

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add mmaivision/datasets/datasets.py tests/test_datasets.py
git commit -m "feat(datasets): 支持 polygon shape_type 并输出 mask 字段

polygon 同时产出 bbox（点集 min/max）和 mask（COCO 多边形格式
List[List[float]]）；rectangle 仍不输出 mask 字段。"
```

---

## Task 7: 未知 label 跳过 + 聚合 warning

**Files:**
- Modify: `mmaivision/datasets/datasets.py` (`_parse_one`, `load_data_list`, `_parse_shape`)
- Test: `tests/test_datasets.py`

- [ ] **Step 1: 追加测试 `test_unknown_label_skipped`**

```python
def test_unknown_label_skipped(caplog):
    import logging
    caplog.set_level(logging.WARNING)
    ds = _make_dataset()
    a = next(d for d in ds.data_list if d['img_id'] == 'IMG_a')
    # 未知 label "unknown_thing" 不应出现在任何 instance
    assert all(i['bbox_label'] in (0, 1) for i in a['instances'])
    # 警告日志应包含 unknown label 计数（至少 1 个）
    msgs = ' '.join(r.getMessage() for r in caplog.records)
    assert 'unknown' in msgs.lower()
```

- [ ] **Step 2: 运行测试确认失败**

Run:
```bash
pytest tests/test_datasets.py::test_unknown_label_skipped -v
```

Expected: FAIL — 未知 label 当前被静默跳过，无 warning。

- [ ] **Step 3: 引入计数器并在 `load_data_list` 末尾汇总**

把 `_parse_shape` 改造为返回更丰富的状态，并在 `_parse_one` 中聚合计数。最简方案：把计数器以 dict 形式从 `load_data_list` 一路传到 `_parse_shape`。

修改 `_parse_shape` 签名与逻辑：

```python
    def _parse_shape(self, shape: Dict[str, Any], classes,
                     counters: Dict[str, int]) -> Optional[Dict[str, Any]]:
        label = shape.get('label')
        if label not in classes:
            counters['unknown_label'] += 1
            return None
        shape_type = shape.get('shape_type')
        if shape_type not in ('rectangle', 'polygon'):
            counters['bad_type'] += 1
            return None
        points = shape.get('points', [])
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
        if not xs or not ys:
            counters['bad_bbox'] += 1
            return None
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        if (x2 - x1) <= 0 or (y2 - y1) <= 0:
            counters['bad_bbox'] += 1
            return None
        inst: Dict[str, Any] = dict(
            bbox=[x1, y1, x2, y2],
            bbox_label=classes.index(label),
            ignore_flag=int(shape.get('difficult', False)),
        )
        if shape_type == 'polygon':
            flat = [coord for xy in zip(xs, ys) for coord in xy]
            inst['mask'] = [flat]
        return inst
```

修改 `_parse_one` 接受并传递 `counters`：

```python
    def _parse_one(self, json_path: str, stem: str, img_dir: str,
                   classes, counters: Dict[str, int]) -> Dict[str, Any]:
        with open(json_path, 'r', encoding='utf-8') as f:
            obj = json.load(f)

        image_path_field = obj.get('imagePath') or f'{stem}.jpg'
        img_basename = Path(image_path_field.replace('\\', '/')).name
        img_path = osp.join(img_dir, img_basename)

        instances: List[Dict[str, Any]] = []
        for shape in obj.get('shapes', []):
            inst = self._parse_shape(shape, classes, counters)
            if inst is not None:
                instances.append(inst)

        return dict(
            img_path=img_path,
            img_id=stem,
            height=int(obj['imageHeight']),
            width=int(obj['imageWidth']),
            instances=instances,
        )
```

修改 `load_data_list` 初始化 counters 并打 warning：

```python
    def load_data_list(self) -> List[Dict[str, Any]]:
        classes = self._metainfo.get('classes')
        if classes is None:
            raise ValueError(
                'classes must be specified via metainfo, e.g. '
                "metainfo=dict(classes=('dc_line', ...))")

        logger = MMLogger.get_current_instance()
        data_list: List[Dict[str, Any]] = []
        counters: Dict[str, int] = dict(
            unknown_label=0, bad_type=0, bad_bbox=0)
        ann_dir = self._resolve_prefix_dir('ann')
        img_dir = self._resolve_prefix_dir('img')

        for stem_or_path in self._iter_ann_file_lines():
            json_path = self._resolve_json_path(stem_or_path, ann_dir)
            stem = Path(json_path).stem
            try:
                data_info = self._parse_one(
                    json_path, stem, img_dir, classes, counters)
            except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
                logger.warning(f'skip {json_path}: {e}')
                continue
            data_list.append(data_info)

        if any(counters.values()):
            logger.warning(
                f'skipped {counters["unknown_label"]} shapes with unknown '
                f'labels, {counters["bad_type"]} with unsupported '
                f'shape_type, {counters["bad_bbox"]} with invalid bbox')
        logger.info(
            f'loaded {len(data_list)} samples from {self.ann_file}')
        return data_list
```

- [ ] **Step 4: 运行测试确认通过**

Run:
```bash
pytest tests/test_datasets.py::test_unknown_label_skipped -v
```

Expected: PASS。

- [ ] **Step 5: 跑回归确保前面用例还过**

Run:
```bash
pytest tests/test_datasets.py -v
```

Expected: 全部 PASS（5 个用例）。

- [ ] **Step 6: 提交**

```bash
git add mmaivision/datasets/datasets.py tests/test_datasets.py
git commit -m "feat(datasets): 聚合统计未知 label / 非法 shape / 非法 bbox

新增 counters 计数器在 _parse_shape 中累计，load_data_list 末尾
打一条汇总 warning，避免逐条刷屏。"
```

---

## Task 8: 非法 shape_type 跳过测试

**Files:**
- Test: `tests/test_datasets.py`

(实现已在 Task 7 完成；本 Task 仅补测试。)

- [ ] **Step 1: 追加测试 `test_invalid_shape_type_skipped`**

```python
def test_invalid_shape_type_skipped(caplog):
    import logging
    caplog.set_level(logging.WARNING)
    ds = _make_dataset()
    a = next(d for d in ds.data_list if d['img_id'] == 'IMG_a')
    # IMG_a 含一个 shape_type='circle' 的 dc_line，应被跳过
    # IMG_a 有效 dc_line 实例：两个 rectangle (含 difficult)
    # + 一个 fuse polygon = 3 个
    assert len(a['instances']) == 3
    msgs = ' '.join(r.getMessage() for r in caplog.records)
    assert 'shape_type' in msgs.lower() or 'unsupported' in msgs.lower()
```

- [ ] **Step 2: 运行测试确认通过**

Run:
```bash
pytest tests/test_datasets.py::test_invalid_shape_type_skipped -v
```

Expected: PASS。

- [ ] **Step 3: 提交**

```bash
git add tests/test_datasets.py
git commit -m "test: 补充非法 shape_type 跳过 + 汇总 warning 的测试"
```

---

## Task 9: 损坏 JSON 跳过不抛

**Files:**
- Test: `tests/test_datasets.py`

(实现已在 Task 4 完成；本 Task 仅补显式测试。)

- [ ] **Step 1: 追加测试 `test_corrupt_json_skipped_not_raised`**

```python
def test_corrupt_json_skipped_not_raised(caplog):
    import logging
    caplog.set_level(logging.WARNING)
    # 构造时不应抛
    ds = _make_dataset()
    ids = {d['img_id'] for d in ds.data_list}
    # IMG_corrupt 不应出现在 data_list 中
    assert 'IMG_corrupt' not in ids
    # 应有警告日志含 IMG_corrupt 路径
    msgs = ' '.join(r.getMessage() for r in caplog.records)
    assert 'IMG_corrupt' in msgs
```

- [ ] **Step 2: 运行测试确认通过**

Run:
```bash
pytest tests/test_datasets.py::test_corrupt_json_skipped_not_raised -v
```

Expected: PASS。

- [ ] **Step 3: 提交**

```bash
git add tests/test_datasets.py
git commit -m "test: 补充损坏 JSON 不抛 + warn 包含路径的测试"
```

---

## Task 10: `min_size` 过滤

**Files:**
- Modify: `mmaivision/datasets/datasets.py` (`load_data_list`, `_parse_shape`)
- Test: `tests/test_datasets.py`

- [ ] **Step 1: 追加测试 `test_min_size_filter`**

```python
def test_min_size_filter():
    # 把 min_size 设大到把 IMG_a 第一个 (10,20,30,40) rect 也过滤掉
    # bbox 宽 = 20, 高 = 20，min_size=25 应该把它丢掉
    ds = _make_dataset(filter_cfg=dict(filter_empty_gt=False, min_size=25))
    a = next(d for d in ds.data_list if d['img_id'] == 'IMG_a')
    for inst in a['instances']:
        x1, y1, x2, y2 = inst['bbox']
        assert (x2 - x1) >= 25 and (y2 - y1) >= 25


def test_min_size_default_keeps_small():
    ds = _make_dataset()  # 默认 min_size=1
    a = next(d for d in ds.data_list if d['img_id'] == 'IMG_a')
    # 默认应保留 3 个有效 instance
    assert len(a['instances']) == 3
```

- [ ] **Step 2: 运行测试确认失败**

Run:
```bash
pytest tests/test_datasets.py::test_min_size_filter tests/test_datasets.py::test_min_size_default_keeps_small -v
```

Expected: `test_min_size_filter` FAIL（当前不读 filter_cfg.min_size，所有 ≥0 的 bbox 都保留）；`test_min_size_default_keeps_small` PASS。

- [ ] **Step 3: 在 `_parse_shape` 中读 min_size 并应用**

修改 `_parse_shape` 签名添加 `min_size`：

```python
    def _parse_shape(self, shape: Dict[str, Any], classes,
                     counters: Dict[str, int],
                     min_size: int) -> Optional[Dict[str, Any]]:
        label = shape.get('label')
        if label not in classes:
            counters['unknown_label'] += 1
            return None
        shape_type = shape.get('shape_type')
        if shape_type not in ('rectangle', 'polygon'):
            counters['bad_type'] += 1
            return None
        points = shape.get('points', [])
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
        if not xs or not ys:
            counters['bad_bbox'] += 1
            return None
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        if (x2 - x1) < min_size or (y2 - y1) < min_size:
            counters['bad_bbox'] += 1
            return None
        inst: Dict[str, Any] = dict(
            bbox=[x1, y1, x2, y2],
            bbox_label=classes.index(label),
            ignore_flag=int(shape.get('difficult', False)),
        )
        if shape_type == 'polygon':
            flat = [coord for xy in zip(xs, ys) for coord in xy]
            inst['mask'] = [flat]
        return inst
```

修改 `_parse_one` 把 min_size 透传：

```python
    def _parse_one(self, json_path: str, stem: str, img_dir: str,
                   classes, counters: Dict[str, int],
                   min_size: int) -> Dict[str, Any]:
        with open(json_path, 'r', encoding='utf-8') as f:
            obj = json.load(f)

        image_path_field = obj.get('imagePath') or f'{stem}.jpg'
        img_basename = Path(image_path_field.replace('\\', '/')).name
        img_path = osp.join(img_dir, img_basename)

        instances: List[Dict[str, Any]] = []
        for shape in obj.get('shapes', []):
            inst = self._parse_shape(shape, classes, counters, min_size)
            if inst is not None:
                instances.append(inst)

        return dict(
            img_path=img_path,
            img_id=stem,
            height=int(obj['imageHeight']),
            width=int(obj['imageWidth']),
            instances=instances,
        )
```

在 `load_data_list` 中从 `self.filter_cfg` 读 `min_size`，注意 `filter_cfg` 可能是 None：

```python
    def load_data_list(self) -> List[Dict[str, Any]]:
        classes = self._metainfo.get('classes')
        if classes is None:
            raise ValueError(
                'classes must be specified via metainfo, e.g. '
                "metainfo=dict(classes=('dc_line', ...))")

        filter_cfg = self.filter_cfg or {}
        min_size = int(filter_cfg.get('min_size', 1))

        logger = MMLogger.get_current_instance()
        data_list: List[Dict[str, Any]] = []
        counters: Dict[str, int] = dict(
            unknown_label=0, bad_type=0, bad_bbox=0)
        ann_dir = self._resolve_prefix_dir('ann')
        img_dir = self._resolve_prefix_dir('img')

        for stem_or_path in self._iter_ann_file_lines():
            json_path = self._resolve_json_path(stem_or_path, ann_dir)
            stem = Path(json_path).stem
            try:
                data_info = self._parse_one(
                    json_path, stem, img_dir, classes, counters, min_size)
            except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
                logger.warning(f'skip {json_path}: {e}')
                continue
            data_list.append(data_info)

        if any(counters.values()):
            logger.warning(
                f'skipped {counters["unknown_label"]} shapes with unknown '
                f'labels, {counters["bad_type"]} with unsupported '
                f'shape_type, {counters["bad_bbox"]} with invalid bbox')
        logger.info(
            f'loaded {len(data_list)} samples from {self.ann_file}')
        return data_list
```

- [ ] **Step 4: 运行两个测试 + 回归**

Run:
```bash
pytest tests/test_datasets.py -v
```

Expected: 全部 PASS（10 个用例）。

- [ ] **Step 5: 提交**

```bash
git add mmaivision/datasets/datasets.py tests/test_datasets.py
git commit -m "feat(datasets): 支持 filter_cfg.min_size 在加载阶段过滤小实例

min_size 默认 1，等价于只过滤宽或高为 0 的非法 bbox；> 1 时丢弃
任意维度小于阈值的实例。"
```

---

## Task 11: `filter_empty_gt` — 重写 `filter_data`

**Files:**
- Modify: `mmaivision/datasets/datasets.py` (新增 `filter_data` 方法)
- Test: `tests/test_datasets.py`

- [ ] **Step 1: 追加两个测试**

```python
def test_filter_empty_gt_train_mode():
    ds = _make_dataset(filter_cfg=dict(filter_empty_gt=True, min_size=1))
    ids = {d['img_id'] for d in ds}
    # IMG_b 无 instance，train 模式应被过滤
    assert 'IMG_b' not in ids
    assert 'IMG_a' in ids


def test_filter_empty_gt_test_mode_keeps_all():
    ds = _make_dataset(
        filter_cfg=dict(filter_empty_gt=True, min_size=1),
        test_mode=True,
    )
    ids = {d['img_id'] for d in ds}
    # test 模式应保留 IMG_b（虽然 0 instance）
    assert 'IMG_b' in ids
    assert 'IMG_a' in ids
```

注意：用 `for d in ds` 迭代 dataset 是经过 `filter_data` 之后的视图；`ds.data_list` 是过滤前的全集。BaseDataset 默认会在 `full_init` 中调 `filter_data` 并赋给 `self.data_address`/`self._fully_initialized`。如果想看过滤后的样本，直接用 `len(ds)` 和 `ds.get_data_info(i)` 即可。这里用 `for d in ds` 配合 `pipeline=[]` 拿到原始 dict。

- [ ] **Step 2: 运行测试确认失败**

Run:
```bash
pytest tests/test_datasets.py::test_filter_empty_gt_train_mode tests/test_datasets.py::test_filter_empty_gt_test_mode_keeps_all -v
```

Expected: `test_filter_empty_gt_train_mode` FAIL（IMG_b 当前未被过滤）；`test_filter_empty_gt_test_mode_keeps_all` PASS。

- [ ] **Step 3: 在 `LabelmeDetDataset` 中重写 `filter_data`**

在类内加入：

```python
    def filter_data(self) -> List[Dict[str, Any]]:
        """根据 filter_cfg 过滤 data_list。

        - test_mode=True 时不过滤
        - filter_cfg['filter_empty_gt']=True 时丢掉 0 instance 的样本
        """
        if self.test_mode:
            return self.data_list

        filter_cfg = self.filter_cfg or {}
        if not filter_cfg.get('filter_empty_gt', True):
            return self.data_list

        return [d for d in self.data_list if len(d['instances']) > 0]
```

- [ ] **Step 4: 运行测试 + 回归**

Run:
```bash
pytest tests/test_datasets.py -v
```

Expected: 全部 PASS（12 个用例）。

- [ ] **Step 5: 提交**

```bash
git add mmaivision/datasets/datasets.py tests/test_datasets.py
git commit -m "feat(datasets): 重写 filter_data 支持 filter_empty_gt

训练模式（test_mode=False）下若 filter_empty_gt=True 则丢弃 0 实例
样本；test 模式始终保留全部样本以便评测覆盖。"
```

---

## Task 12: 更新 `mmaivision/datasets/__init__.py`

**Files:**
- Modify: `mmaivision/datasets/__init__.py`

(让 registry 自动发现机制保持工作，并暴露新类。)

- [ ] **Step 1: 重写 `mmaivision/datasets/__init__.py`**

```python
from .datasets import LabelmeDetDataset
from .transforms import CustomTransform

__all__ = ['LabelmeDetDataset', 'CustomTransform']
```

- [ ] **Step 2: 验证 registry 能 build**

Run:
```bash
python -c "
import mmaivision
from mmaivision.registry import DATASETS
print('LabelmeDetDataset' in DATASETS.module_dict)
"
```

Expected: `True`

- [ ] **Step 3: 跑完整测试套件**

Run:
```bash
pytest tests/test_datasets.py -v
```

Expected: 全部 PASS。

- [ ] **Step 4: 提交**

```bash
git add mmaivision/datasets/__init__.py
git commit -m "feat(datasets): __init__ 导出 LabelmeDetDataset

移除 CustomDataset 占位（datasets.py 重写后已不存在），保留
CustomTransform。"
```

---

## Task 13: 写最小可用的 `configs/_base_/dataset.py`

**Files:**
- Modify: `configs/_base_/dataset.py` (当前为空文件)

- [ ] **Step 1: 写入配置内容**

```python
# Labelme/X-AnyLabeling 风格目标检测数据集示例配置。
# 用户使用时把 data_root / classes / ann_file 替换为自己的值。

dataset_type = 'LabelmeDetDataset'
data_root = 'data/my_dataset'

metainfo = dict(classes=('dc_line', ))

# 最小 pipeline：仅读图。后续可在 _base_/dataset.py 之上叠加 transforms。
# 若安装了 mmdet，可在下游 config 里追加 mmdet.PackDetInputs 等。
train_pipeline = [dict(type='LoadImageFromFile')]
test_pipeline = [dict(type='LoadImageFromFile')]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train.txt',
        data_prefix=dict(img='images', ann='annotations'),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=True, min_size=1),
        pipeline=train_pipeline,
    ))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val.txt',
        data_prefix=dict(img='images', ann='annotations'),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
    ))

test_dataloader = val_dataloader

val_evaluator = dict(type='DumpResults', out_file_path='val_results.pkl')
test_evaluator = val_evaluator
```

注意：`val_evaluator` 用 `DumpResults` 作为占位 evaluator（mmengine 内置）；下游用户接入 mmdet 后可改成 `CocoMetric` 等。

- [ ] **Step 2: 验证 Config 能解析**

Run:
```bash
python -c "
from mmengine.config import Config
cfg = Config.fromfile('configs/_base_/dataset.py')
print('dataset type =', cfg.train_dataloader.dataset.type)
print('classes =', cfg.train_dataloader.dataset.metainfo['classes'])
"
```

Expected:
```
dataset type = LabelmeDetDataset
classes = ('dc_line',)
```

- [ ] **Step 3: 提交**

```bash
git add configs/_base_/dataset.py
git commit -m "feat(configs): 添加 LabelmeDetDataset 最小可用配置示例

train/val/test dataloader 三件套 + DumpResults 占位 evaluator；
pipeline 极简只含 LoadImageFromFile，不引入 mmdet 硬依赖。"
```

---

## Task 14: 写验证 demo `tools/verify_dataset.py`

**Files:**
- Create: `tools/verify_dataset.py`

- [ ] **Step 1: 写入脚本内容**

```python
"""LabelmeDetDataset 验证脚本：
build 配置中的 dataset，打印关键字段并把 bbox/polygon 渲染到图片保存。

用法：
    python tools/verify_dataset.py <config> [--out-dir vis] [--num 5]
                                             [--split train]
"""
import argparse
import os
import os.path as osp
from collections import Counter

from mmengine.config import Config

import mmaivision  # noqa: F401 触发 registry 注册
from mmaivision.registry import DATASETS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='路径到 dataset config (.py)')
    parser.add_argument(
        '--out-dir', default='vis', help='可视化输出目录')
    parser.add_argument(
        '--num', type=int, default=5, help='可视化前 N 个样本')
    parser.add_argument(
        '--split',
        default='train',
        choices=['train', 'val', 'test'],
        help='使用 cfg 中哪个 dataloader 的 dataset')
    return parser.parse_args()


def render(data_info, classes):
    try:
        import cv2
    except ImportError as e:
        raise SystemExit(
            'opencv-python 未安装，请运行 `pip install opencv-python`'
        ) from e
    img = cv2.imread(data_info['img_path'])
    if img is None:
        print(f"  [warn] 读不到图片: {data_info['img_path']}，跳过渲染")
        return None
    for inst in data_info['instances']:
        x1, y1, x2, y2 = [int(round(v)) for v in inst['bbox']]
        is_diff = inst['ignore_flag'] == 1
        color = (128, 128, 128) if is_diff else (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        if 'mask' in inst:
            import numpy as np
            for poly in inst['mask']:
                pts = np.asarray(poly, dtype=np.int32).reshape(-1, 2)
                cv2.polylines(img, [pts], True, (255, 0, 0), 2)

        label = classes[inst['bbox_label']]
        text = f"{label}{' (diff)' if is_diff else ''}"
        cv2.putText(img, text, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    dataloader_key = f'{args.split}_dataloader'
    dataset_cfg = getattr(cfg, dataloader_key).dataset
    # 把 pipeline 临时清空 —— 我们要拿原始 dict 渲染，不让 transforms 改 bbox
    dataset_cfg = dict(dataset_cfg)
    dataset_cfg['pipeline'] = []

    ds = DATASETS.build(dataset_cfg)
    classes = ds.metainfo['classes']

    print(f'=== {args.split} dataset ===')
    print(f'  config           : {args.config}')
    print(f'  type             : {dataset_cfg["type"]}')
    print(f'  classes          : {classes}')
    print(f'  len(ds)          : {len(ds)}')

    label_counter = Counter()
    instance_total = 0
    for d in ds.data_list:
        for inst in d['instances']:
            label_counter[classes[inst['bbox_label']]] += 1
            instance_total += 1
    print(f'  total instances  : {instance_total}')
    print(f'  per-class counts : {dict(label_counter)}')

    os.makedirs(args.out_dir, exist_ok=True)
    print(f'\n=== 前 {args.num} 个样本 ===')
    for i in range(min(args.num, len(ds))):
        info = ds.data_list[i]
        n_inst = len(info['instances'])
        preview_bboxes = [inst['bbox'] for inst in info['instances'][:3]]
        print(f'  [{i}] {info["img_id"]} | {info["width"]}x{info["height"]}'
              f' | {n_inst} instances | bboxes[:3]={preview_bboxes}')
        img = render(info, classes)
        if img is not None:
            out_path = osp.join(args.out_dir, f'{info["img_id"]}.jpg')
            import cv2
            cv2.imwrite(out_path, img)
            print(f'        → 已保存 {out_path}')

    print('\n完成。')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 用单测 fixture 跑一遍验证脚本可执行**

先写一个临时小 config 指向 fixture：

```bash
cat > /tmp/labelme_fixture_cfg.py <<'EOF'
dataset_type = 'LabelmeDetDataset'
data_root = 'tests/data/labelme_sample'
metainfo = dict(classes=('dc_line', 'fuse'))
train_pipeline = [dict(type='LoadImageFromFile')]
train_dataloader = dict(
    batch_size=1, num_workers=0,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train.txt',
        data_prefix=dict(img='images', ann='annotations'),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=False, min_size=1),
        pipeline=train_pipeline,
    ))
EOF
```

Run:
```bash
python tools/verify_dataset.py /tmp/labelme_fixture_cfg.py --out-dir /tmp/labelme_vis --num 2
```

Expected (大致)：
```
=== train dataset ===
  config           : /tmp/labelme_fixture_cfg.py
  type             : LabelmeDetDataset
  classes          : ('dc_line', 'fuse')
  len(ds)          : 2
  total instances  : 3
  per-class counts : {'dc_line': 2, 'fuse': 1}

=== 前 2 个样本 ===
  [0] IMG_a | 1500x1000 | 3 instances | bboxes[:3]=[...]
        [warn] 读不到图片: .../IMG_a.jpg，跳过渲染
  [1] IMG_b | 1200x800 | 0 instances | bboxes[:3]=[]
        [warn] 读不到图片: .../IMG_b.jpg，跳过渲染

完成。
```

（因为 fixture 里的"图片"只是 PNG header 占位字节、不是真图，cv2.imread 会返回 None，所以有 warn —— 这是预期；脚本本身没崩说明逻辑正确。）

- [ ] **Step 3: 清理临时文件**

```bash
rm -f /tmp/labelme_fixture_cfg.py
rm -rf /tmp/labelme_vis
```

- [ ] **Step 4: 提交**

```bash
git add tools/verify_dataset.py
git commit -m "feat(tools): 添加 verify_dataset 可视化验证脚本

从 config 构建 dataset（pipeline 强制清空），打印 len/classes/
per-class instance 计数，用 OpenCV 把 bbox 和 polygon 画到原图保存，
方便肉眼检查 loader 输出是否正确。"
```

---

## Task 15: 手工 smoke test（不走 fixture，走最终入口）

**Files:** 无改动；仅运行验证命令。

- [ ] **Step 1: 验证完整测试套件通过**

Run:
```bash
pytest tests/ -v
```

Expected: 全部 PASS（约 12 个用例），无错误。

- [ ] **Step 2: 验证 `configs/_base_/dataset.py` 可被 `Config.fromfile` + `DATASETS.build` 正常加载**

由于 `configs/_base_/dataset.py` 默认指向 `data/my_dataset`（不存在），我们只验证它的语法和 registry 解析能力。临时把 data_root 指向 fixture：

Run:
```bash
python -c "
from mmengine.config import Config

import mmaivision  # noqa
from mmaivision.registry import DATASETS

cfg = Config.fromfile('configs/_base_/dataset.py')
# 覆盖 data_root / classes / ann_file 指向测试 fixture
cfg.train_dataloader.dataset.data_root = 'tests/data/labelme_sample'
cfg.train_dataloader.dataset.ann_file = 'train.txt'
cfg.train_dataloader.dataset.metainfo = dict(classes=('dc_line', 'fuse'))
cfg.train_dataloader.dataset.pipeline = []
cfg.train_dataloader.dataset.filter_cfg = dict(
    filter_empty_gt=False, min_size=1)

ds = DATASETS.build(cfg.train_dataloader.dataset)
print('len =', len(ds))
print('keys =', sorted(ds.data_list[0].keys()))
print('first 1 instance =', ds.data_list[0]['instances'][:1])
"
```

Expected (大致)：
```
len = 2
keys = ['height', 'img_id', 'img_path', 'instances', 'width']
first 1 instance = [{'bbox': [10.0, 20.0, 30.0, 40.0], 'bbox_label': 0, 'ignore_flag': 0}]
```

- [ ] **Step 3: 验证整个仓库 grep 不到旧 scope**

Run:
```bash
grep -rl "mmengine_template" . --include="*.py" --include="*.cfg" --include="*.in" 2>/dev/null
```

Expected: 输出为空。

- [ ] **Step 4: 验证 `git status` 干净**

Run:
```bash
git status
```

Expected: 工作目录干净，所有改动都已 commit。

- [ ] **Step 5: （无新代码，无需 commit；如果上面验证发现遗漏，回退到对应 Task 修复）**

---

## 实现完成后的状态

- 全部测试通过 (`pytest tests/`)
- `mmaivision/datasets/datasets.py` 含 `LabelmeDetDataset` 完整实现
- `mmaivision/datasets/__init__.py` 导出 `LabelmeDetDataset`
- `configs/_base_/dataset.py` 含可用配置模板
- `tools/verify_dataset.py` 提供可视化验证入口
- 仓库内不存在 `mmengine_template` 字符串（除 docs/spec 等已声明的位置）
- 所有 commit 用中文 message，不含 Co-Authored-By trailer
