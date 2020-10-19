import torch as T
from lpd.metrics.metric_base import MetricBase
from lpd.metrics.categorical_accuracy import CategoricalAccuracy


class CategoricalAccuracyWithLogits(MetricBase):
    def __init__(self):
        self.ca = CategoricalAccuracy()
        
    def __call__(self, y_pred: T.Tensor, y_true: T.Tensor):
        y_pred_softmax = T.softmax(y_pred, dim=1)
        return self.ca(y_pred_softmax, y_true)