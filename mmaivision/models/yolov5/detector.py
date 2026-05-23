"""YOLOv5 单阶段检测器。"""
from mmaivision.registry import MODELS

from ..base.single_stage import SingleStageDetector


@MODELS.register_module()
class YOLOv5Detector(SingleStageDetector):
    """YOLOv5 单阶段检测器。

    复用 SingleStageDetector 的 backbone→neck→bbox_head 串联与 forward 分发,
    本轮仅支持 mode='tensor';loss / predict 待后续模块实现。
    """
