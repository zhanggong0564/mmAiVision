"""YOLOv5Detector:串联 backbone → neck → head 的网络容器。"""
from typing import Optional

from mmengine.model import BaseModel
from torch import Tensor

from mmaivision.registry import MODELS


@MODELS.register_module()
class YOLOv5Detector(BaseModel):
    """串联 backbone → neck → head 的网络容器。

    本轮只实现 mode='tensor' 分支,用于 forward 验证 / 推理 / 可视化中间特征。
    loss / predict 分支待后续 loss + postprocess 模块完成。
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

    def forward(self,
                inputs: Tensor,
                data_samples=None,
                mode: str = 'tensor'):
        if mode == 'tensor':
            return self.bbox_head(self.neck(self.backbone(inputs)))
        raise NotImplementedError(
            f"mode={mode!r} 暂未实现,需等待 loss/postprocess 模块。"
            "本轮仅支持 mode='tensor'。")
