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


class TestC3:
    def test_c3_shape_shortcut(self):
        from mmaivision.models.common import C3
        layer = C3(64, 64, n=2, shortcut=True)
        out = layer(torch.randn(1, 64, 32, 32))
        assert out.shape == (1, 64, 32, 32)

    def test_c3_shape_no_shortcut(self):
        from mmaivision.models.common import C3
        layer = C3(32, 64, n=1, shortcut=False)
        out = layer(torch.randn(1, 32, 16, 16))
        assert out.shape == (1, 64, 16, 16)


class TestSPPF:
    def test_sppf_shape(self):
        from mmaivision.models.common import SPPF
        layer = SPPF(64, 64, k=5)
        out = layer(torch.randn(1, 64, 32, 32))
        assert out.shape == (1, 64, 32, 32)


class TestBackbone:
    def test_backbone_forward_shapes_s(self):
        """yolov5s: deepen=0.33, widen=0.5。"""
        from mmaivision.models.backbone import YOLOv5CSPDarknet
        bb = YOLOv5CSPDarknet(deepen_factor=0.33, widen_factor=0.5)
        feats = bb(torch.randn(2, 3, 640, 640))
        assert len(feats) == 3
        # 基础通道 [64,128,256,512,1024] * 0.5 -> make_divisible/8
        # P3=256*0.5=128, P4=512*0.5=256, P5=1024*0.5=512
        assert feats[0].shape == (2, 128, 80, 80)   # P3, stride 8
        assert feats[1].shape == (2, 256, 40, 40)   # P4, stride 16
        assert feats[2].shape == (2, 512, 20, 20)   # P5, stride 32

    @pytest.mark.parametrize('variant,deepen,widen,expected_c', [
        ('n', 0.33, 0.25, (64, 128, 256)),
        ('s', 0.33, 0.50, (128, 256, 512)),
        ('m', 0.67, 0.75, (192, 384, 768)),
        ('l', 1.00, 1.00, (256, 512, 1024)),
        ('x', 1.33, 1.25, (320, 640, 1280)),
    ])
    def test_backbone_all_variants(self, variant, deepen, widen, expected_c):
        from mmaivision.models.backbone import YOLOv5CSPDarknet
        bb = YOLOv5CSPDarknet(deepen_factor=deepen, widen_factor=widen)
        feats = bb(torch.randn(1, 3, 320, 320))
        assert feats[0].shape[1] == expected_c[0], f'{variant} P3 channels'
        assert feats[1].shape[1] == expected_c[1], f'{variant} P4 channels'
        assert feats[2].shape[1] == expected_c[2], f'{variant} P5 channels'

    def test_backbone_invalid_factor_raises(self):
        from mmaivision.models.backbone import YOLOv5CSPDarknet
        with pytest.raises(ValueError, match='必须 > 0'):
            YOLOv5CSPDarknet(deepen_factor=0, widen_factor=0.5)
        with pytest.raises(ValueError, match='必须 > 0'):
            YOLOv5CSPDarknet(deepen_factor=0.33, widen_factor=-1)

    def test_backbone_invalid_out_indices_raises(self):
        from mmaivision.models.backbone import YOLOv5CSPDarknet
        with pytest.raises(ValueError, match='out_indices'):
            YOLOv5CSPDarknet(deepen_factor=0.33, widen_factor=0.5,
                             out_indices=(0, 1, 2))
