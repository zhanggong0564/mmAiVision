"""Tests for LabelmeDetMetric(自包含 VOC mAP)。"""
import torch


def _ds(pred_bboxes, pred_scores, pred_labels, gt_bboxes, gt_labels):
    return dict(
        pred_instances=dict(
            bboxes=torch.as_tensor(pred_bboxes, dtype=torch.float32).reshape(-1, 4),
            scores=torch.as_tensor(pred_scores, dtype=torch.float32).reshape(-1),
            labels=torch.as_tensor(pred_labels, dtype=torch.int64).reshape(-1)),
        gt_instances=dict(
            bboxes=torch.as_tensor(gt_bboxes, dtype=torch.float32).reshape(-1, 4),
            labels=torch.as_tensor(gt_labels, dtype=torch.int64).reshape(-1)))


def _metric(num_classes=2):
    from mmaivision.evaluation.metrics import LabelmeDetMetric
    return LabelmeDetMetric(num_classes=num_classes,
                            class_names=['line', 'QFU'])


class TestLabelmeDetMetric:
    def test_perfect_prediction_map_1(self):
        m = _metric()
        m.process(None, [_ds(
            pred_bboxes=[[10, 10, 50, 50], [60, 60, 90, 90]],
            pred_scores=[0.9, 0.8], pred_labels=[0, 1],
            gt_bboxes=[[10, 10, 50, 50], [60, 60, 90, 90]],
            gt_labels=[0, 1])])
        out = m.compute_metrics(m.results)
        assert abs(out['mAP'] - 1.0) < 1e-6
        assert abs(out['mAP_50'] - 1.0) < 1e-6
        assert abs(out['AP50_line'] - 1.0) < 1e-6
        assert abs(out['AP50_QFU'] - 1.0) < 1e-6

    def test_no_detection_map_0(self):
        m = _metric()
        m.process(None, [_ds(
            pred_bboxes=torch.zeros(0, 4), pred_scores=[], pred_labels=[],
            gt_bboxes=[[10, 10, 50, 50]], gt_labels=[0])])
        out = m.compute_metrics(m.results)
        # 类 0 有 GT 无预测 → AP=0;类 1 无 GT 被忽略 → mAP=0
        assert out['mAP'] == 0.0

    def test_wrong_localization_below_iou_thr(self):
        m = _metric()
        # 预测框与 GT IoU < 0.5 → 视为 FP,AP=0
        m.process(None, [_ds(
            pred_bboxes=[[100, 100, 140, 140]], pred_scores=[0.9],
            pred_labels=[0], gt_bboxes=[[10, 10, 50, 50]], gt_labels=[0])])
        out = m.compute_metrics(m.results)
        assert out['AP50_line'] == 0.0

    def test_duplicate_pred_one_tp_one_fp(self):
        m = _metric(num_classes=1)
        m.class_names = ['line']
        # 两个都命中同一 GT:一个 TP 一个 FP,AP 仍应为 1.0(recall 达到 1)
        m.process(None, [_ds(
            pred_bboxes=[[10, 10, 50, 50], [11, 11, 51, 51]],
            pred_scores=[0.9, 0.8], pred_labels=[0, 0],
            gt_bboxes=[[10, 10, 50, 50]], gt_labels=[0])])
        out = m.compute_metrics(m.results)
        assert abs(out['AP50_line'] - 1.0) < 1e-6
