from enum import Enum

class CallbackPhase(Enum): 
    ON_TRAIN_BEGIN   = 0
    ON_TRAIN_END     = 1
    ON_EPOCH_BEGIN   = 2
    ON_EPOCH_END     = 3
    ON_BATCH_BEGIN   = 4
    ON_BATCH_END     = 5
    ON_TEST_BEGIN    = 6
    ON_TEST_END      = 7

class TrainerState(Enum):
    """
        The state of the trainer
        TRAIN - when the epoch is in the train mode
        VAL - when the epoch is in the validation mode
        TEST - when calling evaluate(...)
        EXTERNAL - all other states, for example, after valiadaion is done and before the next epoch train has begun
    """
    EXTERNAL = 0
    TRAIN    = 1
    VAL      = 2 
    TEST     = 3

class StatsType(Enum):
    TRAIN    = 0
    VAL      = 1 
    TEST     = 2
    def __str__(self):
        return self.name

class MonitorType(Enum):
    LOSS    = 0
    METRIC  = 1
    def __str__(self):
        return self.name

class MonitorMode(Enum):
    MIN  = 0
    MAX  = 1
    def __str__(self):
        return self.name