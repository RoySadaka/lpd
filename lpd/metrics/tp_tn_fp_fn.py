import torch as T
from lpd.metrics.metric_base import MetricConfusionMatrixBase
from lpd.enums.confusion_matrix_based_metric import ConfusionMatrixBasedMetric as metric

class TruePositives(MetricConfusionMatrixBase):
    """
        Agrs:
            num_classes - The number of classes in the classification
            labels - names of classes, for nice prints, if not provided, the class index will be the label
            predictions_to_classes_convertor - (optional) a function that takes y_pred batch and y_true batch and converts it into class indices batch where
                                               each index represents the chosen class
                                               if None:  
                                                        torch.max with indices will be used for multi-class.
                                                        threshold will be used for binary or multilabel.
            threshold - for binary or multilable classification
    """

    def __init__(self, num_classes, labels=None, predictions_to_classes_convertor=None, threshold=0.5):
        super(TruePositives, self).__init__(num_classes, labels, predictions_to_classes_convertor, threshold)

    def __call__(self, y_pred: T.Tensor, y_true: T.Tensor):
        result_per_class = self.get_stats(metric.TP)
        return result_per_class

class TrueNegatives(MetricConfusionMatrixBase):
    """
        Agrs:
            num_classes - The number of classes in the classification
            labels - names of classes, for nice prints, if not provided, the class index will be the label
            predictions_to_classes_convertor - (optional) a function that takes y_pred batch and y_true batch and converts it into class indices batch where
                                               each index represents the chosen class
                                               if None:  
                                                        torch.max with indices will be used for multi-class.
                                                        threshold will be used for binary or multilabel.
            threshold - for binary or multilable classification
    """

    def __init__(self, num_classes, labels=None, predictions_to_classes_convertor=None, threshold=0.5):
        super(TrueNegatives, self).__init__(num_classes, labels, predictions_to_classes_convertor, threshold)

    def __call__(self, y_pred: T.Tensor, y_true: T.Tensor):
        result_per_class = self.get_stats(metric.TN)
        return result_per_class

class FalseNegatives(MetricConfusionMatrixBase):
    """
        Agrs:
            num_classes - The number of classes in the classification
            labels - names of classes, for nice prints, if not provided, the class index will be the label
            predictions_to_classes_convertor - (optional) a function that takes y_pred batch and y_true batch and converts it into class indices batch where
                                               each index represents the chosen class
                                               if None:  
                                                        torch.max with indices will be used for multi-class.
                                                        threshold will be used for binary or multilabel.
            threshold - for binary or multilable classification
    """

    def __init__(self, num_classes, labels=None, predictions_to_classes_convertor=None, threshold=0.5):
        super(FalseNegatives, self).__init__(num_classes, labels, predictions_to_classes_convertor, threshold)

    def __call__(self, y_pred: T.Tensor, y_true: T.Tensor):
        result_per_class = self.get_stats(metric.FN)
        return result_per_class

class FalsePositives(MetricConfusionMatrixBase):
    """
        Agrs:
            num_classes - The number of classes in the classification
            labels - names of classes, for nice prints, if not provided, the class index will be the label
            predictions_to_classes_convertor - (optional) a function that takes y_pred batch and y_true batch and converts it into class indices batch where
                                               each index represents the chosen class
                                               if None:  
                                                        torch.max with indices will be used for multi-class.
                                                        threshold will be used for binary or multilabel.
            threshold - for binary or multilable classification
    """

    def __init__(self, num_classes, labels=None, predictions_to_classes_convertor=None, threshold=0.5):
        super(FalsePositives, self).__init__(num_classes, labels, predictions_to_classes_convertor, threshold)

    def __call__(self, y_pred: T.Tensor, y_true: T.Tensor):
        result_per_class = self.get_stats(metric.FP)
        return result_per_class