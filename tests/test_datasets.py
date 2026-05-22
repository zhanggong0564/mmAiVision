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
