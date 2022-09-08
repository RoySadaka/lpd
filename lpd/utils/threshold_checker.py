from abc import ABC, abstractmethod
from typing import Union

from torch import Tensor


class ThresholdChecker(ABC):
    """
    Check if the current value is better than the previous best value according to different threshold criteria
    This is an abstract class meant to be inherited by different threshold checkers
    Can also be inherited by the user to create custom threshold checkers
    """
    def __init__(self, threshold: float):
        self.threshold = threshold

    @abstractmethod
    def is_new_value_higher(self, new_value: Union[float, Tensor], old_value: Union[float, Tensor]) -> bool:
        pass

    @abstractmethod
    def is_new_value_lower(self, new_value: Union[float, Tensor], old_value: Union[float, Tensor]) -> bool:
        pass


class AbsoluteThresholdChecker(ThresholdChecker):
    """
    A threshold checker that checks if the difference between the current value and the previous best value
     is greater than or equal to a given threshold

    Args:
        threshold - the threshold to check (must be non-negative)
    """
    def __init__(self, threshold: float = 0.0):
        if threshold < 0:
            raise ValueError(f"Threshold must be non-negative, but got {threshold}")
        super(AbsoluteThresholdChecker, self).__init__(threshold)

    def is_new_value_higher(self, new_value: Union[float, Tensor], old_value: Union[float, Tensor]) -> bool:
        return new_value - old_value > self.threshold

    def is_new_value_lower(self, new_value: Union[float, Tensor], old_value: Union[float, Tensor]) -> bool:
        return old_value - new_value > self.threshold


class RelativeThresholdChecker(ThresholdChecker):
    """
    A threshold checker that checks if the relative difference between the current value and the previous best value
     is greater than or equal to a given threshold

    Args:
        threshold - the threshold to check (must be non-negative)
    """
    def __init__(self, threshold: float = 0.0):
        if threshold < 0:
            raise ValueError(f"Threshold must be non-negative, but got {threshold}")
        super(RelativeThresholdChecker, self).__init__(threshold)

    def is_new_value_higher(self, new_value: Union[float, Tensor], old_value: Union[float, Tensor]) -> bool:
        return (new_value - old_value) / old_value > self.threshold

    def is_new_value_lower(self, new_value: Union[float, Tensor], old_value: Union[float, Tensor]) -> bool:
        return (old_value - new_value) / old_value > self.threshold
