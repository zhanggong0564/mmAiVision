"""YOLOv5 检测数据预处理器:堆叠 / 归一化 / 设置 batch_input_shape。"""
from mmengine.model import ImgDataPreprocessor

from mmaivision.registry import MODELS


@MODELS.register_module()
class YOLOv5DetDataPreprocessor(ImgDataPreprocessor):
    """在 ``ImgDataPreprocessor`` 基础上补 ``batch_input_shape`` metainfo。

    默认配置(mean=0 / std=255 / bgr_to_rgb=True)等价 ultralytics 的
    RGB / 255 归一化。letterbox 已把图缩放到固定正方形,这里只做堆叠 +
    归一化,并把 batch 后的 (H, W) 写回每个 data_sample 的 metainfo。
    """

    def forward(self, data: dict, training: bool = False) -> dict:
        data = super().forward(data, training)
        inputs = data['inputs']
        data_samples = data.get('data_samples')
        if data_samples is not None:
            h, w = int(inputs.shape[-2]), int(inputs.shape[-1])
            for ds in data_samples:
                ds.set_metainfo(dict(batch_input_shape=(h, w)))
        return data
