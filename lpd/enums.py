from enum import Enum

class CallbackPhase(Enum): 
    ON_TRAIN_BEGIN   = 0
    ON_TRAIN_END     = 1
    ON_EPOCH_BEGIN   = 2
    ON_EPOCH_END     = 3
    ON_BATCH_BEGIN   = 4
    ON_BATCH_END     = 5

class State(Enum):
    """
        The state of the trainer
        TRAIN - when the epoch is in the train mode
        VAL - when the epoch is in the validation mode
        TEST - when calling evaluate(...)
        EXTERNAL - all other states, for example, after valiadaion is done and before the next epoch train has begun
    """
    EXTERNAL     = 0
    TRAIN        = 1
    VAL          = 2 
    TEST         = 3
