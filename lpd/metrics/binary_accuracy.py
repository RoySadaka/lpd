import torch as T
from lpd.metrics.metric_base import MetricBase


class BinaryAccuracy(MetricBase):
    def __init__(self):
        pass

    def __call__(self, y_pred: T.Tensor, y_true: T.Tensor):
        assert y_true.size() == y_pred.size()
        pred = y_pred.round().long()
        accuracy = pred.eq(y_true).float().sum() / y_true.numel()
        return accuracy


