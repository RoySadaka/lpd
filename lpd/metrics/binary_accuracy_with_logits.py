import torch as T
from lpd.metrics.metric_base import MetricBase
from lpd.metrics.binary_accuracy import BinaryAccuracy

class BinaryAccuracyWithLogits(MetricBase):
    def __init__(self, threshold=0.5):
        self.ba = BinaryAccuracy(threshold)

    def __call__(self, y_pred: T.Tensor, y_true: T.Tensor):
        y_pred_sigmoid = T.sigmoid(y_pred)
        return self.ba(y_pred_sigmoid, y_true)
