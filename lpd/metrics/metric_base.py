class MetricBase(object):
    def __call__(self, y_pred, y_true):
        raise NotImplementedError('Missing __call__ implementation for metric')