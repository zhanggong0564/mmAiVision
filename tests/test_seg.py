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


def _seg_head(nc=2):
    from mmaivision.models.yolov5.head import YOLOv5SegHead
    return YOLOv5SegHead(num_classes=nc, in_channels=(64, 128, 256),
                         num_masks=32)


def _feats():
    # 输入 640 时 P3/P4/P5 = 80/40/20
    return (torch.randn(2, 64, 80, 80),
            torch.randn(2, 128, 40, 40),
            torch.randn(2, 256, 20, 20))


def _gt_with_masks(n_per_img=(2, 1), hw=(640, 640)):
    from mmengine.structures import InstanceData
    H, W = hw
    out = []
    for n in n_per_img:
        gt = InstanceData()
        boxes, masks = [], []
        for k in range(n):
            x1 = 10 + 30 * k
            boxes.append([x1, x1, x1 + 40, x1 + 50])
            m = torch.zeros(H, W, dtype=torch.uint8)
            m[x1:x1 + 50, x1:x1 + 40] = 1
            masks.append(m)
        gt.bboxes = torch.tensor(boxes, dtype=torch.float32)
        gt.labels = torch.zeros(n, dtype=torch.int64)
        gt.masks = torch.stack(masks)
        out.append(gt)
    return out


class TestSegHead:
    def test_forward_returns_pred_and_proto(self):
        head = _seg_head()
        pred_maps, proto = head(_feats())
        assert len(pred_maps) == 3
        # 每层通道 = na*(nc+5+nm) = 3*(2+5+32) = 117
        assert pred_maps[0].shape == (2, 117, 80, 80)
        assert proto.shape == (2, 32, 160, 160)

    def test_loss_returns_four_terms_and_backward(self):
        head = _seg_head()
        pred_maps, proto = head(_feats())
        losses = head.loss_by_feat(pred_maps, proto, _gt_with_masks(),
                                   [{}, {}])
        assert set(losses) == {'loss_bbox', 'loss_obj', 'loss_cls',
                               'loss_mask'}
        total = sum(losses.values())
        total.backward()  # 不报错即图连通

    def test_predict_outputs_masks(self):
        head = _seg_head()
        head.score_thr = 1e-9  # 强制有输出(绕过 score_thr 过滤,同时满足 >0 约束)
        pred_maps, proto = head(_feats())
        results = head.predict_by_feat(pred_maps, proto, [{}, {}])
        assert len(results) == 2
        r0 = results[0]
        assert hasattr(r0, 'masks')
        if r0.bboxes.shape[0] > 0:
            assert r0.masks.shape[0] == r0.bboxes.shape[0]
            assert r0.masks.shape[1:] == (640, 640)
            assert r0.masks.dtype == torch.bool

    def test_loss_mask_zero_when_no_gt(self):
        # 全 batch 空 GT → loss_mask 应为 0(走 all_masks is None 路径)
        from mmengine.structures import InstanceData
        head = _seg_head()
        pred_maps, proto = head(_feats())
        empty = []
        for _ in range(2):
            g = InstanceData()
            g.bboxes = torch.zeros(0, 4)
            g.labels = torch.zeros(0, dtype=torch.int64)
            g.masks = torch.zeros(0, 640, 640, dtype=torch.uint8)
            empty.append(g)
        losses = head.loss_by_feat(pred_maps, proto, empty, [{}, {}])
        assert float(losses['loss_mask']) == 0.0

    def test_loss_with_mixed_masks_batch(self):
        # 一图有 GT+masks,一图空 GT → 不报错,4 个 loss 齐全且可 backward
        from mmengine.structures import InstanceData
        head = _seg_head()
        pred_maps, proto = head(_feats())
        gts = _gt_with_masks(n_per_img=(2,))  # 第一图 2 个实例
        empty = InstanceData()
        empty.bboxes = torch.zeros(0, 4)
        empty.labels = torch.zeros(0, dtype=torch.int64)
        empty.masks = torch.zeros(0, 640, 640, dtype=torch.uint8)
        gts.append(empty)
        losses = head.loss_by_feat(pred_maps, proto, gts, [{}, {}])
        assert set(losses) == {'loss_bbox', 'loss_obj', 'loss_cls', 'loss_mask'}
        sum(losses.values()).backward()


# ---------------------------------------------------------------------------
# TestSegDetector helpers
# ---------------------------------------------------------------------------

def _seg_detector(nc=2):
    from mmengine.registry import init_default_scope
    from mmaivision.registry import MODELS
    init_default_scope('mmaivision')
    cfg = dict(
        type='YOLOv5SegDetector',
        data_preprocessor=dict(
            type='YOLOv5DetDataPreprocessor',
            mean=[0., 0., 0.], std=[255., 255., 255.],
            bgr_to_rgb=True, pad_size_divisor=32),
        backbone=dict(type='YOLOv5CSPDarknet', deepen_factor=0.33,
                      widen_factor=0.25),
        neck=dict(type='YOLOv5PAFPN', in_channels=(64, 128, 256),
                  out_channels=(64, 128, 256), deepen_factor=0.33,
                  widen_factor=0.25),
        head=dict(type='YOLOv5SegHead', num_classes=nc,
                  in_channels=(64, 128, 256), num_masks=32))
    return MODELS.build(cfg)


def _to_model_inputs(model, batch):
    """走 data_preprocessor 得到 (inputs, data_samples)。"""
    data = model.data_preprocessor(batch, training=True)
    return data['inputs'], data['data_samples']


def _seg_batch():
    from mmengine.structures import BaseDataElement
    inputs = torch.randint(0, 255, (2, 3, 640, 640), dtype=torch.uint8)
    samples = []
    gts = _gt_with_masks()
    for g in gts:
        ds = BaseDataElement()
        ds.gt_instances = g
        ds.set_metainfo(dict(ori_shape=(640, 640), img_shape=(640, 640),
                             scale_factor=(1.0, 1.0),
                             pad_param=np.array([0, 0, 0, 0],
                                                dtype=np.float32)))
        samples.append(ds)
    return dict(inputs=list(inputs), data_samples=samples)


class TestSegDetector:
    def test_loss_step(self):
        model = _seg_detector()
        out = model.loss(*_to_model_inputs(model, _seg_batch()))
        assert 'loss_mask' in out

    def test_predict_step_attaches_masks(self):
        model = _seg_detector()
        model.bbox_head.score_thr = 1e-9
        model.eval()
        results = model.test_step(_seg_batch())
        assert hasattr(results[0], 'pred_instances')
        assert hasattr(results[0].pred_instances, 'masks')
        assert results[0].pred_instances.masks.shape[0] > 0


class TestSegMetric:
    def _ds(self, pred_masks, pred_scores, pred_labels, gt_masks, gt_labels):
        return dict(
            pred_instances=dict(
                masks=torch.as_tensor(np.asarray(pred_masks), dtype=torch.bool),
                scores=torch.as_tensor(pred_scores, dtype=torch.float32),
                labels=torch.as_tensor(pred_labels, dtype=torch.int64)),
            gt_instances=dict(
                masks=torch.as_tensor(np.asarray(gt_masks), dtype=torch.bool),
                labels=torch.as_tensor(gt_labels, dtype=torch.int64)))

    def _mask(self, x1, y1, x2, y2, H=20, W=20):
        m = np.zeros((H, W), dtype=bool)
        m[y1:y2, x1:x2] = True
        return m

    def test_perfect_prediction(self):
        from mmaivision.evaluation.metrics import LabelmeSegMetric
        m = LabelmeSegMetric(num_classes=2, class_names=['line', 'QFU'])
        gt = [self._mask(0, 0, 10, 10), self._mask(12, 12, 18, 18)]
        m.process(None, [self._ds(
            pred_masks=gt, pred_scores=[0.9, 0.8], pred_labels=[0, 1],
            gt_masks=gt, gt_labels=[0, 1])])
        out = m.compute_metrics(m.results)
        assert out['mAP_50'] == 1.0

    def test_no_overlap_zero(self):
        from mmaivision.evaluation.metrics import LabelmeSegMetric
        m = LabelmeSegMetric(num_classes=1)
        m.process(None, [self._ds(
            pred_masks=[self._mask(0, 0, 5, 5)], pred_scores=[0.9],
            pred_labels=[0],
            gt_masks=[self._mask(12, 12, 18, 18)], gt_labels=[0])])
        out = m.compute_metrics(m.results)
        assert out['mAP_50'] == 0.0

    def test_duplicate_prediction_counts_as_fp(self):
        # 2 个 GT,两个预测都命中 GT-A(高分 TP、低分重复记 FP),GT-B 漏检。
        # FP 落在 recall 饱和前 → recall 封顶 0.5,VOC AP = 0.5 < 1。
        # (单 GT 时重复 FP 落在 recall=1.0 之后,经 VOC 精度包络不降 AP,
        #  故此处用 2 GT 才能真正暴露 FP 对 AP 的影响。)
        from mmaivision.evaluation.metrics import LabelmeSegMetric
        m = LabelmeSegMetric(num_classes=1)
        a = self._mask(0, 0, 10, 10)
        b = self._mask(12, 12, 18, 18)
        m.process(None, [self._ds(
            pred_masks=[a, a], pred_scores=[0.9, 0.8],
            pred_labels=[0, 0],
            gt_masks=[a, b], gt_labels=[0, 0])])
        out = m.compute_metrics(m.results)
        assert out['mAP_50'] == 0.5

    def test_class_isolation(self):
        # 预测类别与 GT 类别不同 → 不匹配,AP=0
        from mmaivision.evaluation.metrics import LabelmeSegMetric
        m = LabelmeSegMetric(num_classes=2, class_names=['line', 'QFU'])
        gtm = self._mask(0, 0, 10, 10)
        m.process(None, [self._ds(
            pred_masks=[gtm], pred_scores=[0.9], pred_labels=[1],
            gt_masks=[gtm], gt_labels=[0])])
        out = m.compute_metrics(m.results)
        # class 0 有 GT 无正确预测 → AP0=0;class 1 有预测无 GT → AP1=None 跳过
        assert out['mAP_50'] == 0.0

    def test_iou_threshold_boundary(self):
        # IoU 恰好 0.5 应判为 TP(>= 阈值)。
        # _mask(x1,y1,x2,y2) → m[y1:y2, x1:x2]
        # gt: rows 0:10, cols 0:10 → area=100
        # pred: rows 0:10, cols 0:20 → area=200, inter=100, union=200 → IoU=0.5
        from mmaivision.evaluation.metrics import LabelmeSegMetric
        gt = self._mask(0, 0, 10, 10)
        pred = self._mask(0, 0, 20, 10)
        m = LabelmeSegMetric(num_classes=1, iou_thrs=[0.5])
        m.process(None, [self._ds(
            pred_masks=[pred], pred_scores=[0.9], pred_labels=[0],
            gt_masks=[gt], gt_labels=[0])])
        out = m.compute_metrics(m.results)
        # IoU=0.5 >= 0.5 → TP → AP=1.0
        assert out['mAP_50'] == 1.0


class TestConvertSegMapping:
    def test_seg_target_keys_all_have_prefix(self):
        # 不需要网络:构建 seg 目标模型,验证其每个 state_dict key 都能被
        # PREFIX_MAP ∪ SEG_EXTRA_PREFIX_MAP 的某个前缀匹配到(映射表完整)。
        import importlib.util, os
        spec = importlib.util.spec_from_file_location(
            'convert_ultra',
            os.path.join(os.path.dirname(__file__), '..', 'tools',
                         'convert_ultralytics.py'))
        conv = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(conv)
        target = conv.build_target_model('n', num_classes=80, seg=True,
                                         num_masks=32)
        prefix_map = dict(conv.PREFIX_MAP)
        prefix_map.update(conv.SEG_EXTRA_PREFIX_MAP)
        prefixes = sorted(prefix_map, key=len, reverse=True)
        unmatched = []
        for k in target.state_dict():
            if not any(k == p or k.startswith(p + '.') for p in prefixes):
                unmatched.append(k)
        assert not unmatched, f'无前缀匹配的 key: {unmatched[:10]}'
