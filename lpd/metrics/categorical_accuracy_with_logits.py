import torch as T
from lpd.metrics.metric_base import MetricBase
from lpd.metrics.categorical_accuracy import CategoricalAccuracy


class CategoricalAccuracyWithLogits(MetricBase):
    """
        Same as CategoricalAccuracy, but more explicit about the Logits
    """
    def __init__(self):
        self.ca = CategoricalAccuracy()
        
    def __call__(self, y_pred: T.Tensor, y_true: T.Tensor):
        return self.ca(y_pred, y_true)