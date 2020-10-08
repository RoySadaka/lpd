from enum import Enum, auto

class Phase(Enum):
    IDLE          = auto()
    TRAIN_BEGIN   = auto()
    TRAIN_END     = auto()
    EPOCH_BEGIN   = auto()
    EPOCH_END     = auto()
    BATCH_BEGIN   = auto()
    BATCH_END     = auto()
    TEST_BEGIN    = auto()
    TEST_END      = auto()
    def __str__(self):
        return self.name