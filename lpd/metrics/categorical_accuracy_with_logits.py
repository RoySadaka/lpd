import torch as T
from lpd.metrics.metric_base import MetricBase
from lpd.metrics.categorical_accuracy import CategoricalAccuracy
from lpd.enums.metric_method import MetricMethod


class CategoricalAccuracyWithLogits(MetricBase):
    """
        Same as CategoricalAccuracy, but more explicit about the Logits
    """
    def __init__(self, name='CategoricalAccuracyWithLogits'):
        super(CategoricalAccuracyWithLogits, self).__init__(name=name, metric_method=MetricMethod.MEAN)
        self.ca = CategoricalAccuracy()

    def __call__(self, y_pred: T.Tensor, y_true: T.Tensor):
        return self.ca(y_pred, y_true)