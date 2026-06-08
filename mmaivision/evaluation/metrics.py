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
        """逐图即时匹配:masks 在手时就算 IoU 与贪心 TP/FP,只保留标量结果。

        若像检测那样把整个验证集的稠密 mask 堆在内存里(如 345 图 × ~300 实例
        × 640²),会直接 OOM。这里每张图只保留:各预测的 score/label、各 IoU
        阈值下的 TP 标记,以及逐类 GT 计数。逐图按分数降序贪心匹配与“全局排序后
        贪心”等价(GT 只在本图内与预测竞争),故 mAP 与稠密实现完全一致。
        """
        n_thr = len(self.iou_thrs)
        for ds in data_samples:
            pred = ds['pred_instances']
            gt = ds['gt_instances']
            pred_scores = _to_numpy(pred['scores']).reshape(-1)
            pred_labels = _to_numpy(pred['labels']).reshape(-1)
            pred_masks = _to_numpy(pred['masks']).astype(bool)
            gt_masks = _to_numpy(gt['masks']).astype(bool)
            gt_labels = _to_numpy(gt['labels']).reshape(-1)

            num_pred = pred_scores.shape[0]
            tp = np.zeros((n_thr, num_pred), dtype=bool)
            gt_count = np.bincount(
                gt_labels.astype(np.int64),
                minlength=self.num_classes)[:self.num_classes]

            for cls in range(self.num_classes):
                p_idx = np.nonzero(pred_labels == cls)[0]
                g_sel = gt_labels == cls
                if p_idx.size == 0 or not g_sel.any():
                    continue  # 无该类预测,或无该类 GT(预测全记 FP → tp 保持 False)
                # 该类预测按分数降序(= 全局降序在本图内的投影),逐条贪心匹配
                order = p_idx[np.argsort(-pred_scores[p_idx], kind='stable')]
                ious = _mask_iou_matrix(pred_masks[order], gt_masks[g_sel])
                for ti, thr in enumerate(self.iou_thrs):
                    matched = np.zeros(ious.shape[1], dtype=bool)
                    for rank, gi in enumerate(order):
                        j = int(np.argmax(ious[rank]))
                        if ious[rank, j] >= thr and not matched[j]:
                            tp[ti, gi] = True
                            matched[j] = True
            self.results.append(dict(
                pred_scores=pred_scores.astype(np.float32),
                pred_labels=pred_labels,
                tp=tp,
                gt_count=gt_count.astype(np.int64)))

    def _ap_per_class(self, results: List[dict], cls: int,
                      thr_idx: int) -> Optional[float]:
        """单类、单 IoU 阈值的 mask AP;TP/FP 已在 process 阶段逐图算好。

        汇总各图该类预测的 (score, tp),按分数全局降序累积 PR → VOC AP。
        无该类 GT 时返回 None。
        """
        n_gt = int(sum(int(r['gt_count'][cls]) for r in results))
        if n_gt == 0:
            return None
        scores, tps = [], []
        for r in results:
            sel = r['pred_labels'] == cls
            if sel.any():
                scores.append(r['pred_scores'][sel])
                tps.append(r['tp'][thr_idx][sel])
        if not scores:
            return 0.0
        scores = np.concatenate(scores)
        tp = np.concatenate(tps).astype(np.float32)
        order = np.argsort(-scores, kind='stable')
        tp = tp[order]
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(1.0 - tp)
        recall = tp_cum / n_gt
        precision = tp_cum / np.clip(tp_cum + fp_cum, a_min=1e-7, a_max=None)
        return _voc_ap(recall, precision)

    def compute_metrics(self, results: List[dict]) -> dict:
        logger = MMLogger.get_current_instance()
        per_thr_map = []
        per_class_ap50 = {}
        for ti, thr in enumerate(self.iou_thrs):
            aps = []
            for cls in range(self.num_classes):
                ap = self._ap_per_class(results, cls, ti)
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
