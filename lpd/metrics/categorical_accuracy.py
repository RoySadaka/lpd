import torch as T
from lpd.metrics.metric_base import MetricBase
from lpd.enums.metric_method import MetricMethod


class CategoricalAccuracy(MetricBase):
    def __init__(self):
        super(CategoricalAccuracy, self).__init__(MetricMethod.MEAN)

    def __call__(self, y_pred: T.Tensor, y_true: T.Tensor):
        indices = T.max(y_pred, 1)[1]
        correct = T.eq(indices, y_true).view(-1)
        accuracy = correct.float().sum() / correct.shape[0]
        return accuracy