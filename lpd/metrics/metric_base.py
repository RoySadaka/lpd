from lpd.enums.metric_method import MetricMethod
import torch as T
from lpd.enums.confusion_matrix_based_metric import ConfusionMatrixBasedMetric

class MetricBase(object):
    """
        Args:
        metric_method - from lpd.enums.MetricMethod, use this to dictate how this metric is behaving over the batches,
                        whether its accumulates the MEAN, or the SUM, or taking the LAST value (for example in MetricConfusionMatrixBase)
    """
    def __init__(self, metric_method: MetricMethod):
        self.metric_method = metric_method

    def __call__(self, y_pred: T.Tensor, y_true: T.Tensor):
        raise NotImplementedError('Missing __call__ implementation for metric')


class MetricConfusionMatrixBase(MetricBase):
    def __init__(self, num_classes, labels, predictions_to_classes_convertor, threshold):
        super(MetricConfusionMatrixBase, self).__init__(MetricMethod.LAST)
        self.num_classes = num_classes
        self.labels = labels
        if self.labels and len(self.labels) != num_classes:
            raise ValueError(f'[MetricConfusionMatrixBase] - expecting same number for labels as num_classes, but got num_classes = {num_classes}, and {len(self.labels)} labels')
        self.predictions_to_classes_convertor = predictions_to_classes_convertor
        self.threshold = threshold
        # confusion_matrix_ IS MANAGED INTERNALLY BY TRAINER-STATS VIA _set_confusion_matrix
        self.confusion_matrix_ = None

    def _set_confusion_matrix(self, confusion_matrix):
        """
            INTERNAL USE ONLY!
            the confusion matrix is being handled by TrainerStats, that way there is only one 
            confusion matrix per State (TRAIN/VAL/TEST).
            TrainerStats will inject the most updated confusion matrix here 
        """
        self.confusion_matrix_ = confusion_matrix

    def is_binary(self, y_pred, y_true):
        return self.confusion_matrix_.num_classes == 2

    def get_stats(self, metric: ConfusionMatrixBasedMetric):
        stats = self.confusion_matrix_.get_stats()
        return T.Tensor([stats_per_class[metric] for stats_per_class in stats.values()])

    def get_confusion_matrix(self):
        return self.confusion_matrix_.get_confusion_matrix()
