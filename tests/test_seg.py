"""Tests for YOLOv5 实例分割链路:Proto / SegHead / SegDetector / 数据 / 指标。"""
import numpy as np
import torch


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
        from mmengine.structures import InstanceData
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
        for layer in out:
            assert 'gt_idx' in layer
            assert layer['gt_idx'].shape == layer['img_idx'].shape
            if layer['gt_idx'].numel() > 0:
                assert int(layer['gt_idx'].max()) < 3
                assert int(layer['gt_idx'].min()) >= 0

    def test_empty_has_gt_idx(self):
        a = self._assigner()
        anchors = [torch.tensor([[1.5, 2.0], [2.0, 3.0], [4.0, 5.0]])
                   for _ in range(3)]
        out = a([], anchors, [(80, 80), (40, 40), (20, 20)])
        for layer in out:
            assert layer['gt_idx'].numel() == 0
