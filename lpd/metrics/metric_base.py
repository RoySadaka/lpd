import torch as T

class MetricBase(object):
    def __call__(self, y_pred: T.Tensor, y_true: T.Tensor):
        raise NotImplementedError('Missing __call__ implementation for metric')