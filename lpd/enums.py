from enum import Enum, auto

class CallbackPhase(Enum):
    IDLE          = auto()
    TRAIN_BEGIN   = auto()
    TRAIN_END     = auto()
    EPOCH_BEGIN   = auto()
    EPOCH_END     = auto()
    BATCH_BEGIN   = auto()
    BATCH_END     = auto()
    TEST_BEGIN    = auto()
    TEST_END      = auto()

class TrainerState(Enum):
    """
        The state of the trainer
        TRAIN - when the epoch is in the train mode
        VAL - when the epoch is in the validation mode
        TEST - when calling evaluate(...)
        EXTERNAL - all other states, for example, after valiadaion is done and before the next epoch train has begun
    """
    EXTERNAL = auto()
    TRAIN    = auto()
    VAL      = auto() 
    TEST     = auto()

class StatsType(Enum):
    TRAIN    = auto()
    VAL      = auto() 
    TEST     = auto()
    def __str__(self):
        return self.name

class MonitorType(Enum):
    LOSS    = auto()
    METRIC  = auto()
    def __str__(self):
        return self.name

class MonitorMode(Enum):
    MIN  = auto()
    MAX  = auto()
    def __str__(self):
        return self.name