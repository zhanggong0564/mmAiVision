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
        serialize_data=False,  # 保留 data_list 在内存中，方便测试直接读取
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


def test_missing_image_size_keys_skipped():
    """JSON missing imageHeight/imageWidth should be skipped, not crash."""
    from unittest.mock import patch
    from mmengine.logging import MMLogger

    warnings_logged = []
    original_warning = MMLogger.warning

    def capture_warning(self, msg, *args, **kwargs):
        warnings_logged.append(str(msg))
        return original_warning(self, msg, *args, **kwargs)

    with patch.object(MMLogger, 'warning', capture_warning):
        ds = _make_dataset()

    ids = {d['img_id'] for d in ds.data_list}
    assert 'IMG_no_size' not in ids
    # Warning should mention the file
    msgs = ' '.join(warnings_logged)
    assert 'IMG_no_size' in msgs
