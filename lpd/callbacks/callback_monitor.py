from lpd.callbacks.callback_monitor_result import CallbackMonitorResult
from lpd.enums import Phase, State, MonitorType, MonitorMode, StatsType
from lpd.callbacks.callback_context import CallbackContext
from typing import Union, List, Optional, Dict
from math import inf
import torch

from lpd.utils.threshold_checker import ThresholdChecker, AbsoluteThresholdChecker

class CallbackMonitor:
    """
    Will check if the desired metric improved with support for patience
    Args:
        patience - int or None (will be set to inf) track how many epochs/iterations without improvements in monitoring
                    (negative number will set to inf)
        monitor_type - e.g lpd.enums.MonitorType.LOSS
        stats_type - e.g lpd.enums.StatsType.VAL
        monitor_mode - e.g. lpd.enums.MonitorMode.MIN, will check if the metric decreased, MonitorMode.MAX will check for increase
        metric_name - in case of monitor_mode=lpd.enums.MonitorMode.METRIC, provide metric_name, otherwise, leave it None
        threshold_checker - to check if the criteria was met, if None, AbsoluteThresholdChecker with threshold=0.0 will be used
    """
    def __init__(self, monitor_type: MonitorType,
                        stats_type: StatsType,
                        monitor_mode: MonitorMode,
                        patience: int=None,
                        metric_name: Optional[str]=None,
                        threshold_checker: Optional[ThresholdChecker]=None):
        self.patience = inf if patience is None or patience < 0 else patience
        self.patience_countdown = self.patience
        self.monitor_type = monitor_type
        self.stats_type = stats_type
        self.monitor_mode = monitor_mode
        self.threshold_checker = AbsoluteThresholdChecker(monitor_mode) if threshold_checker is None else threshold_checker
        self.metric_name = metric_name
        self.best = None
        self.previous = None
        self.description = self._get_description()
        self._track_invoked = False

    def _get_description(self):
        desc = f'{self.monitor_mode}_{self.stats_type}_{self.monitor_type}'
        if self.metric_name:
            return desc + f'_{self.metric_name}'
        return desc

    def _get_best(self):
        return self.best

    def track(self, callback_context: CallbackContext) -> CallbackMonitorResult:
        c = callback_context #READABILITY DOWN THE ROAD

        # EXTRACT value_to_consider
        if self.monitor_type == MonitorType.LOSS:

            if self.stats_type == StatsType.TRAIN:
                value_to_consider = c.train_stats.get_loss()
            elif self.stats_type == StatsType.VAL:
                value_to_consider = c.val_stats.get_loss()

        elif self.monitor_type == MonitorType.METRIC:

            if self.stats_type == StatsType.TRAIN:
                metrics_to_consider = c.train_stats.get_metrics()
            elif self.stats_type == StatsType.VAL:
                metrics_to_consider = c.val_stats.get_metrics()

            if self.metric_name not in metrics_to_consider:
                raise ValueError(f'[CallbackMonitor] - monitor_mode={MonitorType.METRIC}, but cant find metric with name {self.metric_name}')
            value_to_consider = metrics_to_consider[self.metric_name]

        if not self._track_invoked:
            if self.monitor_mode == MonitorMode.MIN:
                self.best = -torch.log(torch.zeros_like(value_to_consider))  # [[inf,...,inf]]
            elif self.monitor_mode == MonitorMode.MAX:
                self.best = torch.log(torch.zeros_like(value_to_consider)) # [[-inf,...,-inf]]
            self.previous = self._get_best()
            self._track_invoked = True


        # MONITOR
        self.patience_countdown = max(0, self.patience_countdown - 1)
        change_from_previous = value_to_consider - self.previous
        curr_best = self._get_best()
        change_from_best = value_to_consider - curr_best
        curr_previous = self.previous
        self.previous = value_to_consider
        did_improve = False # UNLESS SAID OTHERWISE
        new_best = curr_best # UNLESS SAID OTHERWISE
        name = self.metric_name if self.metric_name else 'loss'

        if len(value_to_consider.shape) == 0 or  \
           (len(value_to_consider.shape) == 1 and value_to_consider.shape[0] == 1):
            if self.threshold_checker(new_value=value_to_consider, old_value=curr_best):
                did_improve = True
                self.patience_countdown = self.patience
                self.best = new_best = value_to_consider
        else:
            if self.patience != inf:
                raise ValueError("[CallbackMonitor] - can't monitor patience for metric that has multiple values")

        return CallbackMonitorResult(did_improve=did_improve,
                                     new_value=value_to_consider,
                                     prev_value=curr_previous,
                                     new_best=new_best,
                                     prev_best=curr_best,
                                     change_from_previous=change_from_previous,
                                     change_from_best=change_from_best,
                                     patience_left=self.patience_countdown,
                                     description=self.description,
                                     name = name)
