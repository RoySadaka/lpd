import torch as T
from lpd.metrics.metric_base import MetricBase
from lpd.metrics.binary_accuracy import BinaryAccuracy
from lpd.enums.metric_method import MetricMethod

class BinaryAccuracyWithLogits(MetricBase):
    """
        BinaryAccuracyWithLogits is basically BinaryAccuracy with threshold=0.0
        when positive examples are >= 0  and negative examples are < 0
    """
    def __init__(self, threshold=0.0):
        super(BinaryAccuracyWithLogits, self).__init__(MetricMethod.MEAN)
        self.ba = BinaryAccuracy(threshold)


    def __call__(self, y_pred: T.Tensor, y_true: T.Tensor):
        return self.ba(y_pred, y_true)
