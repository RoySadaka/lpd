import torch as T
from lpd.metrics.metric_base import MetricBase


class TopKCategoricalAccuracy(MetricBase):
    """
        Take top k predicted classes from our model and see if the correct class was selected as top k. 
        If it was we say that our model was correct.
    """
    def __init__(self, k):
        self.k = k

    def __call__(self, y_pred: T.Tensor, y_true: T.Tensor):
        indices = T.topk(y_pred, self.k, dim=1)[1]
        expanded_y_true = y_true.view(-1, 1).expand(-1, self.k)
        correct = T.sum(T.eq(indices, expanded_y_true), dim=1)
        accuracy = correct.float().sum() / y_true.numel()
        return accuracy