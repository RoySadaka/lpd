from enum import Enum, auto

class StatsType(Enum):
    TRAIN    = auto()
    VAL      = auto() 
    TEST     = auto()
    def __str__(self):
        return self.name