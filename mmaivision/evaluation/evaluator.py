"""mmaivision 默认 Evaluator。

直接复用 mmengine 标准 ``Evaluator``(``process`` 调 ``metric.process``、
``evaluate`` 调 ``metric.evaluate``),与本仓库基于 mmengine ``BaseMetric``
(``process`` + ``compute_metrics``)实现的指标(如 ``LabelmeDetMetric``)兼容。

注:模板早期版本曾为 mmeval 的 ``add`` / ``compute`` 接口重写过 process/evaluate,
但本项目未安装 mmeval,故回归标准接口。
"""
from mmengine.evaluator import Evaluator as MMEngineEvaluator

from mmaivision.registry import EVALUATOR


@EVALUATOR.register_module()
class Evaluator(MMEngineEvaluator):
    """与 mmengine 标准 BaseMetric 接口一致的 Evaluator。"""
