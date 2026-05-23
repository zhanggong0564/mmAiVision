"""Tests for YOLOv5 network components."""
import pytest
import torch
from torch import nn


class TestConv:
    def test_conv_shape(self):
        from mmaivision.models.common import Conv
        layer = Conv(3, 16, k=3, s=2)
        out = layer(torch.randn(1, 3, 64, 64))
        assert out.shape == (1, 16, 32, 32)
        # BN 存在
        assert isinstance(layer.bn, nn.BatchNorm2d)
        # 激活是 SiLU
        assert isinstance(layer.act, nn.SiLU)

    def test_conv_no_act(self):
        from mmaivision.models.common import Conv
        layer = Conv(3, 16, k=1, s=1, act=False)
        out = layer(torch.randn(1, 3, 8, 8))
        assert out.shape == (1, 16, 8, 8)
        assert isinstance(layer.act, nn.Identity)
