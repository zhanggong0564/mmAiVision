"""单阶段检测器基类:串联 backbone → neck → bbox_head 的通用容器。"""
from typing import Optional

from mmengine.model import BaseModel
from torch import Tensor

from mmaivision.registry import MODELS


class SingleStageDetector(BaseModel):
    """单阶段检测器通用容器。

    子类可重写 ``loss`` / ``predict`` 实现训练 / 推理逻辑;
    ``forward(mode='tensor')`` 默认走通 backbone → neck → bbox_head。
    """

    def __init__(self,
                 backbone: dict,
                 neck: dict,
                 head: dict,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck)
        self.bbox_head = MODELS.build(head)

    def extract_feat(self, inputs: Tensor):
        return self.neck(self.backbone(inputs))

    def forward(self,
                inputs: Tensor,
                data_samples=None,
                mode: str = 'tensor'):
        if mode == 'tensor':
            return self.bbox_head(self.extract_feat(inputs))
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        if mode == 'predict':
            return self.predict(inputs, data_samples)
        raise ValueError(
            f"mode={mode!r} 不支持,可选 'tensor' / 'loss' / 'predict'。")

    def loss(self, inputs: Tensor, data_samples=None):
        feats = self.extract_feat(inputs)
        pred_maps = self.bbox_head(feats)
        batch_gt = [s.gt_instances for s in data_samples]
        batch_metas = [s.metainfo for s in data_samples]
        return self.bbox_head.loss_by_feat(pred_maps, batch_gt, batch_metas)

    def predict(self, inputs: Tensor, data_samples=None):
        feats = self.extract_feat(inputs)
        pred_maps = self.bbox_head(feats)
        if data_samples is not None:
            batch_metas = [s.metainfo for s in data_samples]
        else:
            batch_metas = [dict(batch_input_shape=tuple(inputs.shape[-2:]))
                           ] * inputs.shape[0]
        results_list = self.bbox_head.predict_by_feat(pred_maps, batch_metas)
        # 无 data_samples(纯推理)时直接返回预测列表;否则把预测挂回
        # data_samples 的 pred_instances,供评估器读取 gt/pred 配对。
        if data_samples is None:
            return results_list
        for data_sample, pred in zip(data_samples, results_list):
            data_sample.pred_instances = pred
        return data_samples
