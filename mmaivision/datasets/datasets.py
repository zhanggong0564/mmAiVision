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
