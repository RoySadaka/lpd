from enum import Enum, auto

class ConfusionMatrixBasedMetric(Enum):
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