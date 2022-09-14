from abc import ABC, abstractmethod
from typing import Union

from torch import Tensor

from lpd.enums import MonitorMode


class ThresholdChecker(ABC):
    """
    Check if the current value is better than the previous best value according to different threshold criteria
    This is an abstract class meant to be inherited by different threshold checkers
    Can also be inherited by the user to create custom threshold checkers
    """
    def __init__(self, threshold: float):
        self.threshold = threshold

    @abstractmethod
    def __call__(self, new_value: Union[float, Tensor], old_value: Union[float, Tensor]) -> bool:
        pass


class AbsoluteThresholdChecker(ThresholdChecker):
    """
    A threshold checker that checks if the difference between the current value and the previous best value
     is greater than or equal to a given threshold

    Args:
        monitor_mode: MIN or MAX
        threshold - the threshold to check (must be non-negative)
    """
    def __init__(self, monitor_mode: MonitorMode, threshold: float = 0.0):
        if threshold < 0:
            raise ValueError(f"Threshold must be non-negative, but got {threshold}")
        super(AbsoluteThresholdChecker, self).__init__(threshold)
        self.monitor_mode = monitor_mode

    def _is_new_value_lower(self, new_value: Union[float, Tensor], old_value: Union[float, Tensor]) -> bool:
        return old_value - new_value > self.threshold

    def _is_new_value_higher(self, new_value: Union[float, Tensor], old_value: Union[float, Tensor]) -> bool:
        return new_value - old_value > self.threshold

    def __call__(self, new_value: Union[float, Tensor], old_value: Union[float, Tensor]) -> bool:
        if self.monitor_mode == MonitorMode.MIN:
            return self._is_new_value_lower(new_value, old_value)
        if self.monitor_mode == MonitorMode.MAX:
            return self._is_new_value_higher(new_value, old_value)


class RelativeThresholdChecker(ThresholdChecker):
    """
    A threshold checker that checks if the relative difference between the current value and the previous best value
     is greater than or equal to a given threshold

    Args:
        threshold - the threshold to check (must be non-negative)
    """
    def __init__(self, monitor_mode: MonitorMode, threshold: float = 0.0):
        if threshold < 0:
            raise ValueError(f"Threshold must be non-negative, but got {threshold}")
        super(RelativeThresholdChecker, self).__init__(threshold)
        self.monitor_mode = monitor_mode

    def _is_new_value_lower(self, new_value: Union[float, Tensor], old_value: Union[float, Tensor]) -> bool:
        return (old_value - new_value) / old_value > self.threshold

    def _is_new_value_higher(self, new_value: Union[float, Tensor], old_value: Union[float, Tensor]) -> bool:
        return (new_value - old_value) / old_value > self.threshold

    def __call__(self, new_value: Union[float, Tensor], old_value: Union[float, Tensor]) -> bool:
        if self.monitor_mode == MonitorMode.MIN:
            return self._is_new_value_lower(new_value, old_value)
        if self.monitor_mode == MonitorMode.MAX:
            return self._is_new_value_higher(new_value, old_value)
