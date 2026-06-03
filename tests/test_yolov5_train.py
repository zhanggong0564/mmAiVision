"""Tests for YOLOv5 训练 / 推理 pipeline。"""
import pytest
import torch


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


_ANCHORS = [
    [(10, 13), (16, 30), (33, 23)],
    [(30, 61), (62, 45), (59, 119)],
    [(116, 90), (156, 198), (373, 326)],
]


def _head(num_classes=80):
    from mmaivision.models.yolov5.head import YOLOv5Head
    return YOLOv5Head(
        num_classes=num_classes,
        in_channels=(128, 256, 512),
        prior_generator=dict(
            type='YOLOv5AnchorGenerator',
            base_sizes=_ANCHORS, strides=[8, 16, 32]),
        bbox_coder=dict(type='YOLOv5BBoxCoder'),
        assigner=dict(
            type='YOLOv5BatchAssigner',
            num_classes=num_classes, strides=[8, 16, 32]))


class TestBBoxCoder:
    def _build(self):
        from mmaivision.models.yolov5.task_utils.bbox_coder import (
            YOLOv5BBoxCoder)
        return YOLOv5BBoxCoder()

    def test_decode_known_values(self):
        coder = self._build()
        pred = torch.zeros(1, 1, 1, 1, 4)
        anchor = torch.tensor([[4., 4.]])
        grid_xy = torch.tensor([[[5., 5.]]])
        out = coder.decode(pred, anchor, grid_xy, stride=8)
        assert out.shape == (1, 1, 1, 1, 4)
        assert torch.allclose(out[0, 0, 0, 0],
                              torch.tensor([28., 28., 60., 60.]))

    def test_decode_batched_shape(self):
        coder = self._build()
        pred = torch.randn(2, 3, 20, 20, 4)
        anchor = torch.tensor([[1., 1.], [2., 2.], [3., 3.]])
        grid_xy = torch.zeros(20, 20, 2)
        out = coder.decode(pred, anchor, grid_xy, stride=8)
        assert out.shape == (2, 3, 20, 20, 4)

    def test_encode_decode_roundtrip(self):
        coder = self._build()
        gt_xywh = torch.tensor([[44., 44., 32., 32.]])
        matched_anchor = torch.tensor([[4., 4.]])
        matched_grid_xy = torch.tensor([[5., 5.]])
        coder.encode(gt_xywh, matched_anchor, matched_grid_xy, 8)
        pred = torch.zeros(1, 1, 1, 1, 4)
        decoded = coder.decode(pred, matched_anchor,
                               torch.tensor([[[5., 5.]]]), 8)
        x1, y1, x2, y2 = decoded[0, 0, 0, 0].tolist()
        cxcywh = [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
        assert all(abs(a - b) < 1e-3
                   for a, b in zip(cxcywh, gt_xywh[0].tolist()))


class TestAssigner:
    def _build(self, num_classes=80):
        from mmaivision.models.yolov5.task_utils.assigner import (
            YOLOv5BatchAssigner)
        return YOLOv5BatchAssigner(
            num_classes=num_classes, strides=[8, 16, 32])

    def _anchors_per_layer(self):
        return [
            torch.tensor([[1.25, 1.625], [2., 3.75], [4.125, 2.875]]),
            torch.tensor([[1.875, 3.8125], [3.875, 2.8125], [3.6875, 7.4375]]),
            torch.tensor([[3.625, 2.8125], [4.875, 6.1875], [11.65, 10.18]]),
        ]

    def test_assigner_basic_match(self):
        from mmengine.structures import InstanceData
        assigner = self._build()
        gt = InstanceData(
            bboxes=torch.tensor([[160., 160., 240., 240.]]),
            labels=torch.tensor([5]))
        assignments = assigner([gt], self._anchors_per_layer(),
                               featmap_sizes=[(80, 80), (40, 40), (20, 20)])
        assert len(assignments) == 3
        total = sum(a['img_idx'].numel() for a in assignments)
        assert total > 0

    def test_assigner_empty_batch_gt(self):
        from mmengine.structures import InstanceData
        assigner = self._build()
        empty = InstanceData(
            bboxes=torch.zeros(0, 4),
            labels=torch.zeros(0, dtype=torch.int64))
        assignments = assigner([empty, empty], self._anchors_per_layer(),
                               featmap_sizes=[(80, 80), (40, 40), (20, 20)])
        assert len(assignments) == 3
        for a in assignments:
            for key in ('img_idx', 'anchor_idx', 'grid_y', 'grid_x',
                        'gt_xy', 'gt_wh', 'gt_class'):
                assert key in a and a[key].numel() == 0

    def test_assigner_invalid_args_raises(self):
        from mmaivision.models.yolov5.task_utils.assigner import (
            YOLOv5BatchAssigner)
        with pytest.raises(ValueError):
            YOLOv5BatchAssigner(num_classes=0, strides=[8, 16, 32])
        with pytest.raises(ValueError):
            YOLOv5BatchAssigner(num_classes=80, strides=[0])


class TestCIoU:
    def test_ciou_identical_boxes(self):
        from mmaivision.models.yolov5.iou_loss import bbox_ciou
        pred = torch.tensor([[10., 10., 50., 50.]])
        target = torch.tensor([[10., 10., 50., 50.]])
        ciou = bbox_ciou(pred, target)
        assert ciou.shape == (1,)
        assert torch.allclose(ciou, torch.tensor([1.0]), atol=1e-5)

    def test_ciou_disjoint_boxes(self):
        from mmaivision.models.yolov5.iou_loss import bbox_ciou
        pred = torch.tensor([[0., 0., 10., 10.]])
        target = torch.tensor([[100., 100., 110., 110.]])
        assert bbox_ciou(pred, target).item() < 0


def _pred_maps(B=2, nc=80, na=3, grad=False):
    return [
        torch.randn(B, na * (nc + 5), 80, 80, requires_grad=grad),
        torch.randn(B, na * (nc + 5), 40, 40, requires_grad=grad),
        torch.randn(B, na * (nc + 5), 20, 20, requires_grad=grad),
    ]


class TestHeadLossByFeat:
    def test_loss_by_feat_basic(self):
        from mmengine.structures import InstanceData
        head = _head()
        pred_maps = _pred_maps(grad=True)
        batch_gt = [
            InstanceData(
                bboxes=torch.tensor([[10., 20., 100., 200.],
                                     [50., 50., 150., 250.]]),
                labels=torch.tensor([0, 1])),
            InstanceData(
                bboxes=torch.tensor([[200., 300., 400., 500.]]),
                labels=torch.tensor([2])),
        ]
        metas = [dict(batch_input_shape=(640, 640))] * 2
        losses = head.loss_by_feat(pred_maps, batch_gt, metas)
        assert set(losses) == {'loss_bbox', 'loss_obj', 'loss_cls'}
        for v in losses.values():
            assert torch.isfinite(v).all() and v.item() >= 0
        sum(losses.values()).backward()
        assert pred_maps[0].grad.abs().sum().item() > 0

    def test_loss_by_feat_all_empty_gt(self):
        from mmengine.structures import InstanceData
        head = _head()
        pred_maps = _pred_maps()
        empty = InstanceData(
            bboxes=torch.zeros(0, 4),
            labels=torch.zeros(0, dtype=torch.int64))
        metas = [dict(batch_input_shape=(640, 640))] * 2
        losses = head.loss_by_feat(pred_maps, [empty, empty], metas)
        assert losses['loss_bbox'].item() == 0.0
        assert losses['loss_cls'].item() == 0.0
        assert losses['loss_obj'].item() > 0

    def test_loss_by_feat_batch_mismatch_raises(self):
        from mmengine.structures import InstanceData
        head = _head()
        pred_maps = _pred_maps(B=2)
        batch_gt = [InstanceData(bboxes=torch.zeros(0, 4),
                                 labels=torch.zeros(0, dtype=torch.int64))]
        metas = [dict(batch_input_shape=(640, 640))]
        with pytest.raises(AssertionError):
            head.loss_by_feat(pred_maps, batch_gt, metas)


class TestHeadPredictByFeat:
    def test_predict_by_feat_returns_instancedata(self):
        from mmengine.structures import InstanceData
        head = _head()
        head.score_thr = 0.01
        B = 2
        pred_maps = _pred_maps(B=B)
        preds = head.predict_by_feat(
            pred_maps, [dict(batch_input_shape=(640, 640))] * B)
        assert isinstance(preds, list) and len(preds) == B
        for p in preds:
            assert isinstance(p, InstanceData)
            assert p.bboxes.ndim == 2 and p.bboxes.shape[1] == 4
            assert p.scores.ndim == 1
            assert p.labels.ndim == 1 and p.labels.dtype == torch.int64
            assert p.bboxes.shape[0] == p.scores.shape[0] == p.labels.shape[0]


class TestDetectorEndToEnd:
    def _model(self):
        from mmaivision.registry import MODELS
        return MODELS.build(dict(
            type='YOLOv5Detector',
            backbone=dict(type='YOLOv5CSPDarknet',
                          deepen_factor=0.33, widen_factor=0.5),
            neck=dict(type='YOLOv5PAFPN',
                      in_channels=(128, 256, 512),
                      out_channels=(128, 256, 512),
                      deepen_factor=0.33, widen_factor=0.5),
            head=dict(type='YOLOv5Head', num_classes=80,
                      in_channels=(128, 256, 512))))

    def _data(self, B=2):
        from mmengine.structures import BaseDataElement, InstanceData
        inputs = torch.randn(B, 3, 640, 640)
        samples = []
        for boxes, labels in [
                ([[10., 20., 100., 200.], [50., 50., 150., 250.]], [0, 1]),
                ([[200., 300., 400., 500.]], [2])]:
            s = BaseDataElement()
            s.gt_instances = InstanceData(
                bboxes=torch.tensor(boxes), labels=torch.tensor(labels))
            s.set_metainfo(dict(batch_input_shape=(640, 640)))
            samples.append(s)
        return inputs, samples[:B]

    def test_detector_loss_mode_runs(self):
        model = self._model()
        inputs, samples = self._data(B=2)
        losses = model.forward(inputs, samples, mode='loss')
        assert set(losses) == {'loss_bbox', 'loss_obj', 'loss_cls'}
        sum(losses.values()).backward()
        grad = model.backbone.stem.conv.weight.grad
        assert grad is not None and grad.abs().sum().item() > 0

    def test_detector_predict_mode_returns_instancedata(self):
        from mmengine.structures import InstanceData
        model = self._model()
        inputs, samples = self._data(B=2)
        preds = model.forward(inputs, samples, mode='predict')
        assert isinstance(preds, list) and len(preds) == 2
        assert all(isinstance(p, InstanceData) for p in preds)

    def test_detector_unknown_mode_raises(self):
        model = self._model()
        inputs, samples = self._data(B=1)
        with pytest.raises(ValueError, match='mode'):
            model.forward(inputs, samples, mode='wrong')
