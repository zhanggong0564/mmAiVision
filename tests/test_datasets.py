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


def test_difficult_sets_ignore_flag():
    ds = _make_dataset()
    a = next(d for d in ds.data_list if d['img_id'] == 'IMG_a')
    flags = sorted(
        i['ignore_flag'] for i in a['instances'] if i['bbox_label'] == 0)
    # IMG_a 有两个 dc_line rectangle：一个 difficult=False, 一个 difficult=True
    # (第三个角点重合的 rect 因 bbox 非法已被丢弃)
    assert flags == [0, 1]


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


def test_unknown_label_skipped():
    """未知 label 应被跳过，并在加载结束时输出汇总 warning。"""
    from unittest.mock import patch

    with patch('mmengine.logging.MMLogger.warning') as mock_warn:
        ds = _make_dataset()
    a = next(d for d in ds.data_list if d['img_id'] == 'IMG_a')
    # 未知 label "unknown_thing" 不应出现在任何 instance
    assert all(i['bbox_label'] in (0, 1) for i in a['instances'])
    # 汇总 warning 应包含 unknown label 计数
    msgs = ' '.join(call.args[0] for call in mock_warn.call_args_list)
    assert 'unknown' in msgs.lower()


def test_invalid_shape_type_skipped():
    """shape_type 既不是 rectangle 也不是 polygon 的 shape 应被跳过，
    并在汇总 warning 中体现。"""
    from unittest.mock import patch

    with patch('mmengine.logging.MMLogger.warning') as mock_warn:
        ds = _make_dataset()
    a = next(d for d in ds.data_list if d['img_id'] == 'IMG_a')
    # IMG_a 含一个 shape_type='circle' 的 dc_line，应被跳过
    # IMG_a 有效实例：两个 rectangle (含 difficult) + 一个 fuse polygon = 3 个
    assert len(a['instances']) == 3
    msgs = ' '.join(call.args[0] for call in mock_warn.call_args_list)
    assert 'shape_type' in msgs.lower() or 'unsupported' in msgs.lower()
