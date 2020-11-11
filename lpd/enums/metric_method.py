from enum import Enum, auto

class MetricMethod(Enum):
    """
        use this to dictate how a metric behaves over the batches,
        whether its accumulates the MEAN, or the SUM, or taking the LAST value (for example in MetricConfusionMatrixBase)
    """
    MEAN    = auto()
    SUM     = auto() 
    LAST    = auto() 
    
    def __str__(self):
        return self.name