from enum import Enum, auto

class MonitorMode(Enum):
    MIN  = auto()
    MAX  = auto()
    def __str__(self):
        return self.name