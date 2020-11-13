from enum import Enum, auto

class ConfusionMatrixBasedMetric(Enum):
    """
        An enum that represents all kinds of metric names that can be obtained from 
        a metric that is based on MetricConfusionMatrixBase
    """
    TP                          = auto()
    FP                          = auto()
    FN                          = auto()
    TN                          = auto()
    PRECISION                   = auto()
    SENSITIVITY                 = auto()
    SPECIFICITY                 = auto()
    RECALL                      = auto()
    POSITIVE_PREDICTIVE_VALUE   = auto()
    NEGATIVE_PREDICTIVE_VALUE   = auto()
    ACCURACY                    = auto()
    F1SCORE                     = auto()

    def __str__(self):
        return self.name