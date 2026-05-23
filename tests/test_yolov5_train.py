"""Tests for YOLOv5 训练 / 推理 pipeline。"""
import pytest
import torch
from torch import nn


class TestAnchorGenerator:
    def _build(self):
        from mmaivision.models.yolov5.task_utils.prior_generator import (
            YOLOv5AnchorGenerator)
        return YOLOv5AnchorGenerator(
            base_sizes=[
                [(10, 13), (16, 30), (33, 23)],
                [(30, 61), (62, 45), (59, 119)],
                [(116, 90), (156, 198), (373, 326)],
            ],
            strides=[8, 16, 32],
        )

    def test_grid_priors_shape(self):
        gen = self._build()
        priors = gen.grid_priors([(80, 80), (40, 40), (20, 20)])
        assert len(priors) == 3
        assert priors[0].shape == (3, 80, 80, 2)
        assert priors[1].shape == (3, 40, 40, 2)
        assert priors[2].shape == (3, 20, 20, 2)
        # base_sizes[0][0] / strides[0] = (10/8, 13/8)
        assert torch.allclose(priors[0][0, 0, 0],
                              torch.tensor([10 / 8, 13 / 8]))

    def test_grid_xy_shape_and_values(self):
        gen = self._build()
        grids = gen.grid_xy([(4, 5), (2, 3), (1, 2)])
        assert len(grids) == 3
        # 第 0 层:(ny=4, nx=5, 2)
        assert grids[0].shape == (4, 5, 2)
        assert torch.equal(grids[0][0, 0], torch.tensor([0., 0.]))
        # nx-1 = 4, ny-1 = 3
        assert torch.equal(grids[0][3, 4], torch.tensor([4., 3.]))

    def test_invalid_strides_raises(self):
        from mmaivision.models.yolov5.task_utils.prior_generator import (
            YOLOv5AnchorGenerator)
        with pytest.raises(ValueError, match='stride'):
            YOLOv5AnchorGenerator(
                base_sizes=[[(10, 13), (16, 30), (33, 23)]],
                strides=[0])
