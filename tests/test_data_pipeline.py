"""Tests for YOLOv5 检测数据管线:transforms + data_preprocessor。"""
import numpy as np
import torch


def _sample_results():
    return dict(
        img=np.zeros((480, 640, 3), dtype=np.uint8),
        img_shape=(480, 640),
        ori_shape=(480, 640),
        img_id='IMG_x',
        img_path='/tmp/IMG_x.jpg',
        instances=[
            dict(bbox=[10., 20., 110., 220.], bbox_label=0, ignore_flag=0),
            dict(bbox=[300., 100., 500., 400.], bbox_label=1, ignore_flag=0),
        ])


class TestLoadLabelmeAnnotations:
    def test_builds_gt_arrays(self):
        from mmaivision.datasets.transforms import LoadLabelmeAnnotations
        out = LoadLabelmeAnnotations().transform(_sample_results())
        assert out['gt_bboxes'].shape == (2, 4)
        assert out['gt_bboxes'].dtype == np.float32
        assert out['gt_bboxes_labels'].tolist() == [0, 1]

    def test_empty_instances(self):
        from mmaivision.datasets.transforms import LoadLabelmeAnnotations
        res = _sample_results()
        res['instances'] = []
        out = LoadLabelmeAnnotations().transform(res)
        assert out['gt_bboxes'].shape == (0, 4)
        assert out['gt_bboxes_labels'].shape == (0,)


class TestLetterResize:
    def test_letterbox_shape_and_bbox_scaling(self):
        from mmaivision.datasets.transforms import (LetterResize,
                                                    LoadLabelmeAnnotations)
        res = LoadLabelmeAnnotations().transform(_sample_results())
        out = LetterResize(scale=640).transform(res)
        # 480x640 → r = 640/640 = 1.0,但高需 pad 到 640
        assert out['img'].shape == (640, 640, 3)
        assert out['scale_factor'] == (1.0, 1.0)
        top = out['pad_param'][0]
        # 原 bbox y 应整体下移 top
        assert np.allclose(out['gt_bboxes'][0, 1], 20.0 + top)
        # 所有 bbox 落在 [0, 640]
        assert out['gt_bboxes'].min() >= 0
        assert out['gt_bboxes'].max() <= 640


class TestPackDetInputs:
    def test_pack(self):
        from mmaivision.datasets.transforms import (LetterResize,
                                                    LoadLabelmeAnnotations,
                                                    PackDetInputs)
        res = LoadLabelmeAnnotations().transform(_sample_results())
        res = LetterResize(scale=640).transform(res)
        out = PackDetInputs().transform(res)
        assert out['inputs'].shape == (3, 640, 640)
        assert out['inputs'].dtype == torch.uint8
        ds = out['data_samples']
        assert ds.gt_instances.bboxes.shape == (2, 4)
        assert ds.gt_instances.labels.dtype == torch.int64
        assert 'img_id' in ds.metainfo and 'scale_factor' in ds.metainfo


class TestDataPreprocessor:
    def test_normalize_and_batch_input_shape(self):
        from mmengine.dataset import pseudo_collate

        from mmaivision.datasets.transforms import (LetterResize,
                                                    LoadLabelmeAnnotations,
                                                    PackDetInputs)
        from mmaivision.registry import MODELS

        pre = MODELS.build(dict(
            type='YOLOv5DetDataPreprocessor',
            mean=[0., 0., 0.], std=[255., 255., 255.],
            bgr_to_rgb=True, pad_size_divisor=32))

        def make():
            res = LoadLabelmeAnnotations().transform(_sample_results())
            res = LetterResize(scale=640).transform(res)
            return PackDetInputs().transform(res)

        batch = pseudo_collate([make(), make()])
        data = pre(batch, training=True)
        assert data['inputs'].shape == (2, 3, 640, 640)
        assert data['inputs'].dtype == torch.float32
        assert 0.0 <= float(data['inputs'].min())
        assert float(data['inputs'].max()) <= 1.0
        for ds in data['data_samples']:
            assert ds.metainfo['batch_input_shape'] == (640, 640)
