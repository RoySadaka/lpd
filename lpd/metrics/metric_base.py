from lpd.enums.metric_method import MetricMethod
import torch
from lpd.enums.confusion_matrix_based_metric import ConfusionMatrixBasedMetric
from lpd.metrics.confusion_matrix import ConfusionMatrix

class MetricBase:
    """
        Args:
        metric_method - from lpd.enums.MetricMethod, use this to dictate how this metric is behaving over the batches,
                        whether its accumulates the MEAN, or the SUM, or taking the LAST value (for example in MetricConfusionMatrixBase)
    """
    def __init__(self, name: str, metric_method: MetricMethod):
        self.name = name
        self.metric_method = metric_method

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        raise NotImplementedError('Missing __call__ implementation for metric')


class MetricConfusionMatrixBase(MetricBase):
    """
        _INJECTED_CONFUSION_MATRIX is for INTERNAL USE ONLY!
        the confusion matrix is being handled by TrainerStats, that way there is only one 
        confusion matrix per State (TRAIN/VAL/TEST).
        TrainerStats will inject the most updated confusion matrix here 
    """
    _INJECTED_CONFUSION_MATRIX: ConfusionMatrix = None

    def __init__(self, name, num_classes, labels, predictions_to_classes_convertor, threshold):
        super(MetricConfusionMatrixBase, self).__init__(name=name, metric_method=MetricMethod.LAST)
        self.num_classes = num_classes
        self.labels = labels
        if self.labels and len(self.labels) != num_classes:
            raise ValueError(f'[{self.name}] - expecting same number for labels as num_classes, but got num_classes = {num_classes}, and {len(self.labels)} labels')
        self.predictions_to_classes_convertor = predictions_to_classes_convertor
        self.threshold = threshold

    def _is_binary(self):
        return MetricConfusionMatrixBase._INJECTED_CONFUSION_MATRIX.num_classes == 2

    def get_stats(self, metric: ConfusionMatrixBasedMetric):
        stats = MetricConfusionMatrixBase._INJECTED_CONFUSION_MATRIX.get_stats()
        result_per_class = torch.Tensor([stats_per_class[metric] for stats_per_class in stats.values()])
        if self._is_binary():
            return result_per_class[1]
        return result_per_class

    def get_confusion_matrix(self):
        return MetricConfusionMatrixBase._INJECTED_CONFUSION_MATRIX.get_confusion_matrix()
