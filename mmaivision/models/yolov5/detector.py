"""YOLOv5 单阶段检测器。"""
from mmaivision.registry import MODELS

from ..base.single_stage import SingleStageDetector


@MODELS.register_module()
class YOLOv5Detector(SingleStageDetector):
    """YOLOv5 单阶段检测器。

    复用 SingleStageDetector 的 backbone→neck→bbox_head 串联与 forward 分发,
    本轮仅支持 mode='tensor';loss / predict 待后续模块实现。
    """


@MODELS.register_module()
class YOLOv5SegDetector(SingleStageDetector):
    """YOLOv5 实例分割检测器。

    与 ``YOLOv5Detector`` 的区别:head 前向返回 ``(pred_maps, proto)``,
    loss / predict 解包后把 proto 传给 head 的对应方法。
    """

    def loss(self, inputs, data_samples=None):
        feats = self.extract_feat(inputs)
        pred_maps, proto = self.bbox_head(feats)
        batch_gt = [s.gt_instances for s in data_samples]
        batch_metas = [s.metainfo for s in data_samples]
        return self.bbox_head.loss_by_feat(
            pred_maps, proto, batch_gt, batch_metas)

    def predict(self, inputs, data_samples=None):
        feats = self.extract_feat(inputs)
        pred_maps, proto = self.bbox_head(feats)
        if data_samples is not None:
            batch_metas = [s.metainfo for s in data_samples]
        else:
            batch_metas = [dict(batch_input_shape=tuple(inputs.shape[-2:]))
                           ] * inputs.shape[0]
        results_list = self.bbox_head.predict_by_feat(
            pred_maps, proto, batch_metas)
        if data_samples is None:
            return results_list
        for data_sample, pred in zip(data_samples, results_list):
            data_sample.pred_instances = pred
        return data_samples
