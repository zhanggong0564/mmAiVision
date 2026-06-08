"""检测评估指标。

``LabelmeDetMetric`` 是自包含的 VOC 风格 mAP 实现(不依赖 mmdet / pycocotools),
基于 mmengine 标准 ``BaseMetric``(``process`` + ``compute_metrics``),与 Runner
默认 Evaluator 直接兼容。预测与 GT 均在模型输入(letterbox)坐标系下比较。
"""
from typing import List, Optional, Sequence

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmaivision.registry import METRICS


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _bbox_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """``(N, M)`` IoU 矩阵,xyxy。"""
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float32)
    area1 = ((boxes1[:, 2] - boxes1[:, 0]) *
             (boxes1[:, 3] - boxes1[:, 1]))[:, None]
    area2 = ((boxes2[:, 2] - boxes2[:, 0]) *
             (boxes2[:, 3] - boxes2[:, 1]))[None, :]
    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.clip(rb - lt, a_min=0, a_max=None)
    inter = wh[..., 0] * wh[..., 1]
    return inter / np.clip(area1 + area2 - inter, a_min=1e-7, a_max=None)


def _voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """全点插值 AP(VOC2010+ / COCO 风格,PR 曲线下面积)。"""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _mask_iou_matrix(masks1: np.ndarray, masks2: np.ndarray) -> np.ndarray:
    """``(N, M)`` mask IoU,输入均为 bool ``(K, H, W)``。"""
    if masks1.shape[0] == 0 or masks2.shape[0] == 0:
        return np.zeros((masks1.shape[0], masks2.shape[0]), dtype=np.float32)
    a = masks1.reshape(masks1.shape[0], -1).astype(np.float32)
    b = masks2.reshape(masks2.shape[0], -1).astype(np.float32)
    inter = a @ b.T
    area1 = a.sum(axis=1)[:, None]
    area2 = b.sum(axis=1)[None, :]
    union = np.clip(area1 + area2 - inter, a_min=1e-7, a_max=None)
    return inter / union


@METRICS.register_module()
class LabelmeDetMetric(BaseMetric):
    """VOC 风格检测 mAP 指标。

    Args:
        num_classes: 类别数。
        iou_thrs: IoU 阈值列表,mAP 在其上取均值,默认 ``[0.5]``。
        class_names: 类别名(用于逐类 AP 的日志键),可选。
        prefix: 指标名前缀。
    """

    default_prefix = 'labelme'

    def __init__(self,
                 num_classes: int,
                 iou_thrs: Sequence[float] = (0.5, ),
                 class_names: Optional[Sequence[str]] = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        if num_classes < 1:
            raise ValueError(f'num_classes 必须 >= 1, got {num_classes}')
        self.num_classes = num_classes
        self.iou_thrs = list(iou_thrs)
        self.class_names = list(class_names) if class_names else [
            f'class_{i}' for i in range(num_classes)]

    def process(self, data_batch, data_samples: List[dict]) -> None:
        """每个 data_sample 已被 Evaluator 转成 dict(含 gt/pred_instances)。"""
        for ds in data_samples:
            pred = ds['pred_instances']
            gt = ds['gt_instances']
            self.results.append(dict(
                pred_bboxes=_to_numpy(pred['bboxes']).reshape(-1, 4),
                pred_scores=_to_numpy(pred['scores']).reshape(-1),
                pred_labels=_to_numpy(pred['labels']).reshape(-1),
                gt_bboxes=_to_numpy(gt['bboxes']).reshape(-1, 4),
                gt_labels=_to_numpy(gt['labels']).reshape(-1)))

    def _ap_per_class(self, results: List[dict], cls: int,
                      iou_thr: float) -> Optional[float]:
        """单类、单 IoU 阈值的 AP。无该类 GT 时返回 None。"""
        # 收集该类全部预测:(img_idx, score, bbox)
        preds = []
        n_gt = 0
        gt_by_img = []
        for img_idx, r in enumerate(results):
            gm = r['gt_labels'] == cls
            gt_b = r['gt_bboxes'][gm]
            n_gt += gt_b.shape[0]
            gt_by_img.append(dict(bboxes=gt_b,
                                  matched=np.zeros(gt_b.shape[0], dtype=bool)))
            pm = r['pred_labels'] == cls
            for b, s in zip(r['pred_bboxes'][pm], r['pred_scores'][pm]):
                preds.append((img_idx, float(s), b))
        if n_gt == 0:
            return None
        if not preds:
            return 0.0

        preds.sort(key=lambda x: x[1], reverse=True)
        tp = np.zeros(len(preds), dtype=np.float32)
        fp = np.zeros(len(preds), dtype=np.float32)
        for i, (img_idx, _, box) in enumerate(preds):
            g = gt_by_img[img_idx]
            if g['bboxes'].shape[0] == 0:
                fp[i] = 1
                continue
            ious = _bbox_iou_matrix(box[None, :], g['bboxes'])[0]
            j = int(np.argmax(ious))
            if ious[j] >= iou_thr and not g['matched'][j]:
                tp[i] = 1
                g['matched'][j] = True
            else:
                fp[i] = 1

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / n_gt
        precision = tp_cum / np.clip(tp_cum + fp_cum, a_min=1e-7, a_max=None)
        return _voc_ap(recall, precision)

    def compute_metrics(self, results: List[dict]) -> dict:
        logger = MMLogger.get_current_instance()
        # ap_table[thr_idx][cls] = AP 或 None
        per_thr_map = []
        per_class_ap50 = {}
        for ti, thr in enumerate(self.iou_thrs):
            aps = []
            for cls in range(self.num_classes):
                ap = self._ap_per_class(results, cls, thr)
                if ap is None:
                    continue
                aps.append(ap)
                if abs(thr - 0.5) < 1e-6:
                    per_class_ap50[self.class_names[cls]] = ap
            per_thr_map.append(float(np.mean(aps)) if aps else 0.0)

        metrics = {}
        metrics['mAP'] = float(np.mean(per_thr_map)) if per_thr_map else 0.0
        if any(abs(t - 0.5) < 1e-6 for t in self.iou_thrs):
            i = [abs(t - 0.5) < 1e-6 for t in self.iou_thrs].index(True)
            metrics['mAP_50'] = per_thr_map[i]
        for name, ap in per_class_ap50.items():
            metrics[f'AP50_{name}'] = ap

        logger.info('mAP=%.4f  %s' % (
            metrics['mAP'],
            '  '.join(f'{k}={v:.4f}' for k, v in per_class_ap50.items())))
        return metrics


@METRICS.register_module()
class LabelmeSegMetric(BaseMetric):
    """VOC 风格实例分割 mask mAP 指标(自包含,不依赖 pycocotools)。

    与 ``LabelmeDetMetric`` 结构一致,把 bbox IoU 换成 mask IoU。pred 与 gt
    mask 均在模型输入(letterbox)坐标系下比较。
    """

    default_prefix = 'labelme_seg'

    def __init__(self,
                 num_classes: int,
                 iou_thrs: Sequence[float] = (0.5, ),
                 class_names: Optional[Sequence[str]] = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        if num_classes < 1:
            raise ValueError(f'num_classes 必须 >= 1, got {num_classes}')
        self.num_classes = num_classes
        self.iou_thrs = list(iou_thrs)
        self.class_names = list(class_names) if class_names else [
            f'class_{i}' for i in range(num_classes)]

    def process(self, data_batch, data_samples: List[dict]) -> None:
        for ds in data_samples:
            pred = ds['pred_instances']
            gt = ds['gt_instances']
            self.results.append(dict(
                pred_masks=_to_numpy(pred['masks']).astype(bool),
                pred_scores=_to_numpy(pred['scores']).reshape(-1),
                pred_labels=_to_numpy(pred['labels']).reshape(-1),
                gt_masks=_to_numpy(gt['masks']).astype(bool),
                gt_labels=_to_numpy(gt['labels']).reshape(-1)))

    def _ap_per_class(self, results: List[dict], cls: int,
                      iou_thr: float) -> Optional[float]:
        preds = []
        n_gt = 0
        gt_by_img = []
        for img_idx, r in enumerate(results):
            gm = r['gt_labels'] == cls
            gt_m = r['gt_masks'][gm]
            n_gt += gt_m.shape[0]
            gt_by_img.append(dict(
                masks=gt_m, matched=np.zeros(gt_m.shape[0], dtype=bool)))
            pm = r['pred_labels'] == cls
            for mask, s in zip(r['pred_masks'][pm], r['pred_scores'][pm]):
                preds.append((img_idx, float(s), mask))
        if n_gt == 0:
            return None
        if not preds:
            return 0.0

        preds.sort(key=lambda x: x[1], reverse=True)
        tp = np.zeros(len(preds), dtype=np.float32)
        fp = np.zeros(len(preds), dtype=np.float32)
        for i, (img_idx, _, mask) in enumerate(preds):
            g = gt_by_img[img_idx]
            if g['masks'].shape[0] == 0:
                fp[i] = 1
                continue
            ious = _mask_iou_matrix(mask[None, ...], g['masks'])[0]
            j = int(np.argmax(ious))
            if ious[j] >= iou_thr and not g['matched'][j]:
                tp[i] = 1
                g['matched'][j] = True
            else:
                fp[i] = 1

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / n_gt
        precision = tp_cum / np.clip(tp_cum + fp_cum, a_min=1e-7, a_max=None)
        return _voc_ap(recall, precision)

    def compute_metrics(self, results: List[dict]) -> dict:
        logger = MMLogger.get_current_instance()
        per_thr_map = []
        per_class_ap50 = {}
        for thr in self.iou_thrs:
            aps = []
            for cls in range(self.num_classes):
                ap = self._ap_per_class(results, cls, thr)
                if ap is None:
                    continue
                aps.append(ap)
                if abs(thr - 0.5) < 1e-6:
                    per_class_ap50[self.class_names[cls]] = ap
            per_thr_map.append(float(np.mean(aps)) if aps else 0.0)

        metrics = {}
        metrics['mAP'] = float(np.mean(per_thr_map)) if per_thr_map else 0.0
        if any(abs(t - 0.5) < 1e-6 for t in self.iou_thrs):
            i = [abs(t - 0.5) < 1e-6 for t in self.iou_thrs].index(True)
            metrics['mAP_50'] = per_thr_map[i]
        for name, ap in per_class_ap50.items():
            metrics[f'AP50_{name}'] = ap

        logger.info('seg mAP=%.4f  %s' % (
            metrics['mAP'],
            '  '.join(f'{k}={v:.4f}' for k, v in per_class_ap50.items())))
        return metrics
