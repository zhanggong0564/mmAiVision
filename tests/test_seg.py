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
