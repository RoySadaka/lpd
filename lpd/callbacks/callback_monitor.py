from lpd.enums import Phase, State, MonitorType, MonitorMode, StatsType
from lpd.callbacks.callback_context import CallbackContext
from typing import Union, List, Optional, Dict
from math import inf

class CallbackMonitor():
    """
    Will check if the desired metric improved with support for patience
    Agrs:
        patience - int or None (will set to inf), track how many invocations without improvements
        monitor_type - e.g lpd.enums.MonitorType.LOSS
        stats_type - e.g lpd.enums.StatsType.VAL
        monitor_mode - e.g. lpd.enums.MonitorMode.MIN, min wothh check if the metric decreased, MAX will check for increase
        metric_name - in case of monitor_mode=lpd.enums.MonitorMode.METRIC, provide metric_name, otherwise, leave it None
    """
    def __init__(self, patience: int, monitor_type: MonitorType, stats_type: StatsType, monitor_mode: MonitorMode, metric_name: Optional[str]=None):
        self.patience = patience if patience else inf
        self.patience_countdown = self.patience
        self.monitor_type = monitor_type
        self.stats_type = stats_type
        self.monitor_mode = monitor_mode
        self.metric_name = metric_name
        self.minimum = inf
        self.maximum = -inf
        self.previous = self._get_best()
        self.description = self._get_description()

    def _get_description(self):
        desc = f'{self.monitor_mode}_{self.stats_type}_{self.monitor_type}'
        if self.metric_name:
            return desc + f'_{self.metric_name}'
        return desc

    def _get_best(self):
        return self.minimum if self.monitor_mode == MonitorMode.MIN else self.maximum

    def track(self, callback_context: CallbackContext):
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

        # MONITOR
        self.patience_countdown = max(0, self.patience_countdown - 1)
        change_from_previous = value_to_consider - self.previous
        curr_best = self._get_best()
        change_from_best = value_to_consider - curr_best
        curr_minimum = self.minimum
        curr_maximum = self.maximum
        self.minimum = min(self.minimum, value_to_consider)
        self.maximum = max(self.maximum, value_to_consider)
        curr_previous = self.previous
        self.previous = value_to_consider
        did_improve = False
        new_best = self._get_best()

        if  self.monitor_mode == MonitorMode.MIN and value_to_consider < curr_minimum or \
            self.monitor_mode == MonitorMode.MAX and value_to_consider > curr_maximum:
            did_improve = True
            self.patience_countdown = self.patience
        
        return CallbackMonitorResult(did_improve=did_improve, 
                                     new_value=value_to_consider, 
                                     prev_value=curr_previous,
                                     new_best=new_best,
                                     prev_best=curr_best,
                                     change_from_previous=change_from_previous,
                                     change_from_best=change_from_best,
                                     patience_left=self.patience_countdown, 
                                     description=self.description)


class CallbackMonitorResult():
    def __init__(self, did_improve: bool, 
                        new_value: float, 
                        prev_value: float,
                        new_best: float,
                        prev_best: float,
                        change_from_previous: float,
                        change_from_best: float,
                        patience_left: int,
                        description: str):
        self.did_improve = did_improve
        self.new_value = new_value
        self.prev_value = prev_value
        self.new_best = new_best
        self.prev_best = prev_best
        self.change_from_previous = change_from_previous
        self.change_from_best = change_from_best
        self.patience_left = patience_left
        self.description = description

    def has_improved(self):
        return self.did_improve

    def has_patience(self):
        return self.patience_left > 0
