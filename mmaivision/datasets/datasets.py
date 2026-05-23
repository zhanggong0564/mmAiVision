"""LabelmeDetDataset: 加载 X-AnyLabeling / Labelme 风格 JSON 标注的目标检测数据集。"""
import json
import os.path as osp
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

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
            except (FileNotFoundError, json.JSONDecodeError, OSError,
                    KeyError) as e:
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

    def _resolve_prefix_dir(self, key: str) -> str:
        """Return the directory under data_prefix[key].

        BaseDataset._join_prefix in __init__ already joined this value
        with data_root, so we just return it as-is. Fall back to
        data_root if the key is absent.
        """
        raw = (self.data_prefix or {}).get(key, '')
        return raw or self.data_root

    def _iter_ann_file_lines(self) -> Iterator[str]:
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
                   classes: Sequence[str],
                   counters: Dict[str, int],
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

    def _parse_shape(self, shape: Dict[str, Any], classes: Sequence[str],
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
