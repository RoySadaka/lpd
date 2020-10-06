from enum import Enum, auto

class CallbackPhase(Enum): 
    ON_TRAIN_BEGIN   = auto()
    ON_TRAIN_END     = auto()
    ON_EPOCH_BEGIN   = auto()
    ON_EPOCH_END     = auto()
    ON_BATCH_BEGIN   = auto()
    ON_BATCH_END     = auto()
    ON_TEST_BEGIN    = auto()
    ON_TEST_END      = auto()

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