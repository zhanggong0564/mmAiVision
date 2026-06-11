"""自定义 Hook。"""

from mmengine.dataset import Compose
from mmengine.hooks import Hook

from mmaivision.registry import HOOKS


@HOOKS.register_module()
class PipelineSwitchHook(Hook):
    """到指定 epoch 切换训练 pipeline,用于 YOLOv5 的 ``close_mosaic``。

    Mosaic / MixUp 会制造大量截断、合成目标,训练分布与真实分布有偏差;YOLOv5
    的做法是在最后若干 epoch **关闭马赛克类增强**,换成只含 ``LetterResize`` 的
    干净 pipeline,让模型在接近真实分布上收敛。本 Hook 在 ``switch_epoch``(从 0
    计)开始把训练集 pipeline 替换为 ``switch_pipeline``。

    用法(配置)::

        custom_hooks = [
            dict(type='PipelineSwitchHook',
                 switch_epoch=max_epochs - 15,
                 switch_pipeline=train_pipeline_stage2),
        ]

    其中 ``train_pipeline_stage2`` 通常为 ``[LoadImageFromFile,
    LoadLabelmeAnnotations, LetterResize, RandomFlip, PackDetInputs]``(无
    Mosaic/MixUp)。

    Args:
        switch_epoch: 切换发生的 epoch(0 起),即从该 epoch 起用新 pipeline。
        switch_pipeline: 新的 transform 配置列表。
    """

    def __init__(self, switch_epoch: int, switch_pipeline: list):
        if switch_epoch < 0:
            raise ValueError(f'switch_epoch 必须 >= 0, got {switch_epoch}')
        self.switch_epoch = switch_epoch
        self.switch_pipeline = switch_pipeline
        self._restart_dataloader = False
        self._has_switched = False

    def before_train_epoch(self, runner) -> None:
        """到点替换 dataset.pipeline;persistent_workers 时强制重建 iterator。"""
        epoch = runner.epoch
        train_loader = runner.train_dataloader
        if epoch >= self.switch_epoch and not self._has_switched:
            runner.logger.info(
                f'Switch train pipeline at epoch {epoch} '
                f'(close mosaic/mixup).')
            train_loader.dataset.pipeline = Compose(self.switch_pipeline)
            # persistent_workers=True 时旧 worker 持有旧 pipeline 副本,
            # 需让 DataLoader 在本 epoch 重新初始化 worker 才能生效。
            if hasattr(train_loader, 'persistent_workers') \
                    and train_loader.persistent_workers:
                train_loader._DataLoader__initialized = False
                train_loader._iterator = None
                self._restart_dataloader = True
            self._has_switched = True
        elif self._restart_dataloader:
            # 切换完成后恢复标志,避免后续 epoch 反复重建。
            train_loader._DataLoader__initialized = True
