"""YOLOv5 模型族:backbone / neck / head / detector + task_utils。"""
from . import task_utils  # noqa: F401  触发 task_utils 注册
from .backbone import YOLOv5CSPDarknet
from .detector import YOLOv5Detector
from .head import YOLOv5Head
from .neck import YOLOv5PAFPN

__all__ = [
    'YOLOv5CSPDarknet',
    'YOLOv5PAFPN',
    'YOLOv5Head',
    'YOLOv5Detector',
]
