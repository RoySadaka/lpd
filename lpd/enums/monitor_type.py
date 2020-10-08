from enum import Enum, auto

class MonitorType(Enum):
    LOSS    = auto()
    METRIC  = auto()
    def __str__(self):
        return self.name