from enum import Enum, auto

class State(Enum):
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
    PREDICT  = auto()
    def __str__(self):
        return self.name