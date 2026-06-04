"""Tests for YOLOv5 实例分割链路:Proto / SegHead / SegDetector / 数据 / 指标。"""
import numpy as np
import torch
from mmengine.structures import InstanceData


class TestProto:
    def test_proto_shape(self):
        from mmaivision.models.yolov5.common import Proto
        # P3: stride 8 → 输入 640 时特征 80×80;Proto 上采样 ×2 → 160×160
        proto = Proto(c1=64, c_=256, c2=32)
        out = proto(torch.randn(2, 64, 80, 80))
        assert out.shape == (2, 32, 160, 160)


class TestAssignerGtIdx:
    def _assigner(self):
        from mmaivision.models.yolov5.task_utils.assigner import \
            YOLOv5BatchAssigner
        return YOLOv5BatchAssigner(num_classes=2, strides=[8, 16, 32])

    def test_outputs_gt_idx(self):
        a = self._assigner()
        gt0 = InstanceData()
        gt0.bboxes = torch.tensor([[10., 10., 50., 50.],
                                   [60., 60., 120., 140.]])
        gt0.labels = torch.tensor([0, 1])
        gt1 = InstanceData()
        gt1.bboxes = torch.tensor([[20., 30., 80., 90.]])
        gt1.labels = torch.tensor([1])
        anchors = [torch.tensor([[1.5, 2.0], [2.0, 3.0], [4.0, 5.0]])
                   for _ in range(3)]
        sizes = [(80, 80), (40, 40), (20, 20)]
        out = a([gt0, gt1], anchors, sizes)
        T = 3  # gt0(2) + gt1(1) 共 3 个 GT 实例（按图序拼接）
        flat_classes = torch.tensor([0, 1, 1])  # gt0.labels + gt1.labels
        for layer in out:
            assert 'gt_idx' in layer
            assert layer['gt_idx'].shape == layer['img_idx'].shape
            gi = layer['gt_idx']
            if gi.numel() > 0:
                assert int(gi.max()) < T
                assert int(gi.min()) >= 0
                # 验证 gt_idx 确实指向正确的 GT 实例（类别一致性）
                assert torch.equal(layer['gt_class'], flat_classes[gi])

    def test_empty_has_gt_idx(self):
        a = self._assigner()
        anchors = [torch.tensor([[1.5, 2.0], [2.0, 3.0], [4.0, 5.0]])
                   for _ in range(3)]
        out = a([], anchors, [(80, 80), (40, 40), (20, 20)])
        for layer in out:
            assert layer['gt_idx'].numel() == 0


class TestPolygonToMask:
    def _results_with_polygon(self):
        # 一张 100×100 图,一个三角形 polygon(label 0)+ 一个纯 rectangle(label 1)
        return dict(
            img=np.zeros((100, 100, 3), dtype=np.uint8),
            img_shape=(100, 100),
            ori_shape=(100, 100),
            img_id='P0',
            img_path='/tmp/P0.jpg',
            instances=[
                dict(bbox=[10., 10., 50., 50.], bbox_label=0, ignore_flag=0,
                     mask=[[10., 10., 50., 10., 10., 50.]]),
                dict(bbox=[60., 60., 90., 90.], bbox_label=1, ignore_flag=0),
            ])

    def test_load_builds_polygons(self):
        from mmaivision.datasets.transforms import LoadLabelmeAnnotations
        out = LoadLabelmeAnnotations().transform(self._results_with_polygon())
        polys = out['gt_polygons']
        assert len(polys) == 2
        assert polys[0].shape == (3, 2)      # 三角形 3 点
        assert polys[1].shape == (0, 2)      # 无 polygon → 空

    def test_letterresize_scales_polygons(self):
        from mmaivision.datasets.transforms import (LetterResize,
                                                    LoadLabelmeAnnotations)
        res = LoadLabelmeAnnotations().transform(self._results_with_polygon())
        out = LetterResize(scale=200).transform(res)
        # 100→200 等比 ×2,无 padding(正方形),点坐标翻倍
        assert np.allclose(out['gt_polygons'][0][0], [20., 20.], atol=1e-3)

    def test_letterresize_polygon_with_padding(self):
        from mmaivision.datasets.transforms import (LetterResize,
                                                    LoadLabelmeAnnotations)
        # 100(H)×50(W) 图,scale=100 → r=min(100/100,100/50)=1.0,
        # 缩放后 100×50,水平居中 pad:left=(100-50)//2=25,top=0
        res = dict(
            img=np.zeros((100, 50, 3), dtype=np.uint8),
            img_shape=(100, 50), ori_shape=(100, 50),
            img_id='P1', img_path='/tmp/P1.jpg',
            instances=[dict(bbox=[10., 10., 40., 40.], bbox_label=0,
                            ignore_flag=0,
                            mask=[[10., 10., 40., 10., 10., 40.]])])
        out = LetterResize(scale=100).transform(
            LoadLabelmeAnnotations().transform(res))
        # 点 (10,10) → ×1.0 + (left=25, top=0) = (35, 10)
        assert np.allclose(out['gt_polygons'][0][0], [35., 10.], atol=1e-3)

    def test_pack_rasterizes_masks(self):
        from mmaivision.datasets.transforms import (LetterResize, PackDetInputs,
                                                    LoadLabelmeAnnotations)
        res = LoadLabelmeAnnotations().transform(self._results_with_polygon())
        res = LetterResize(scale=100).transform(res)
        out = PackDetInputs().transform(res)
        masks = out['data_samples'].gt_instances.masks
        assert masks.shape == (2, 100, 100)
        assert masks.dtype == torch.uint8
        # 三角形 mask 内部有像素,纯 rectangle 实例 mask 为空(全 0)
        assert masks[0].sum() > 0
        assert masks[1].sum() == 0
