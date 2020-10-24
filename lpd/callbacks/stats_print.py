from lpd.enums import Phase, State, MonitorType, MonitorMode, StatsType
from lpd.callbacks.callback_base import CallbackBase
from lpd.callbacks.callback_context import CallbackContext
from lpd.callbacks.callback_monitor import CallbackMonitor, CallbackMonitorResult
from lpd.utils.torch_utils import get_lrs_from_optimizer
from typing import Union, List, Optional, Dict, Iterable

class StatsPrint(CallbackBase):
    """
        Prints informative summary of the trainer stats including loss and metrics.
        Args:
            apply_on_phase - see in CallbackBase
            apply_on_states - see in CallbackBase
            round_values_on_print_to - see in CallbackBase

            train_metrics_monitors - CallbackMonitor instance or list of CallbackMonitor instances that will track the improvement of the trainer's metrics.
                                    if empty list, no monitoring will be applied to metrics.
                                    if None, it will assign CallbackMonitor with MonitorMode.MAX per each metric

                                    val_metric_monitors will be applied automatically based in train_metrics_monitors
    """

    def __init__(self, apply_on_phase: Phase=Phase.EPOCH_END, 
                       apply_on_states: Union[State, List[State]]=State.EXTERNAL, 
                       round_values_on_print_to: Optional[int]=None,
                       train_metrics_monitors: Union[CallbackMonitor,Iterable[CallbackMonitor]]=None):
        super(StatsPrint, self).__init__(apply_on_phase, apply_on_states, round_values_on_print_to)
        self.train_loss_monitor = CallbackMonitor(None, MonitorType.LOSS, StatsType.TRAIN, MonitorMode.MIN)
        self.val_loss_monitor = CallbackMonitor(None, MonitorType.LOSS, StatsType.VAL, MonitorMode.MIN)
        self.train_metrics_monitors = self._parse_train_metrics_monitors(train_metrics_monitors)
        self.val_metric_monitors = None
        self._validate_stats_type()
        self.GREEN_PRINT_COLOR = "\033[92m"
        self.END_PRINT_COLOR = "\033[0m"

    def _parse_train_metrics_monitors(self, train_metrics_monitors):
        if isinstance(train_metrics_monitors, list):
            return train_metrics_monitors
        if isinstance(train_metrics_monitors, CallbackMonitor):
            return [train_metrics_monitors]

    def _validate_stats_type(self):
        if self.train_metrics_monitors is not None:
            for m in self.train_metrics_monitors:
                if m.stats_type != StatsType.TRAIN:
                    raise ValueError(f'[StatsPrint] - train_metrics_monitors contains monitor with stats_Type {m.stats_Type}, expected {StatsType.TRAIN}')
            
    def _ensure_metrics_created(self, callback_context: CallbackContext):
        if self.train_metrics_monitors is not None and self.val_metric_monitors is not None:
            # ENSURED ALREADY
            return

        metric_names = callback_context.trainer.metric_name_to_func.keys()
        if self.train_metrics_monitors is None:
            self.train_metrics_monitors = [CallbackMonitor(None, MonitorType.METRIC, StatsType.TRAIN, MonitorMode.MAX, metric_name) for metric_name in metric_names]

        self.val_metric_monitors = []
        for m in self.train_metrics_monitors:
            self.val_metric_monitors.append(CallbackMonitor(m.patience, m.monitor_type, StatsType.VAL, m.monitor_mode, m.metric_name))

    def _get_print_from_monitor_result(self, monitor_result: CallbackMonitorResult) -> str:
        r = self.round_to #READABILITY
        mtr = monitor_result #READABILITY
        return f'curr:{r(mtr.new_value)}, prev:{r(mtr.prev_value)}, best:{r(mtr.new_best)}, change_from_prev:{r(mtr.change_from_previous)}, change_from_best:{r(mtr.change_from_best)}'

    def _get_print_from_metrics(self, train_metric_monitor_results: Iterable[CallbackMonitorResult], prefix: str='') -> str:
        gdim = self._get_did_improved_colored #READABILITY 

        if len(train_metric_monitor_results) == 0:
            return 'no metrics found'
        prints = []
        pre = ''
        for monitor_result in train_metric_monitor_results:
            name = monitor_result.name
            prints.append(f'{pre}name: {name}{gdim(monitor_result)}, {self._get_print_from_monitor_result(monitor_result)}')
            pre = prefix # APPEND PREFIX FROM THE SECOND METRIC AND ON

        return '\n'.join(prints)

    def _get_did_improved_colored(self, monitor_result):
        if monitor_result.has_improved():
            return self.GREEN_PRINT_COLOR + ' IMPROVED' + self.END_PRINT_COLOR
        return ' '

    def __call__(self, callback_context: CallbackContext):
        c = callback_context #READABILITY 
        r = self.round_to #READABILITY
        gdim = self._get_did_improved_colored #READABILITY 
        gmfp = self._get_print_from_metrics #READABILITY 

        self._ensure_metrics_created(c)

        # INVOKE MONITORS
        t_loss_monitor_result = self.train_loss_monitor.track(c)
        v_loss_monitor_result = self.val_loss_monitor.track(c)

        train_metric_monitor_results = [m.track(c) for m in self.train_metrics_monitors]
        val_metric_monitor_results = [m.track(c) for m in self.val_metric_monitors]

        current_lrs = get_lrs_from_optimizer(c.trainer.optimizer)

        print('------------------------------------------------------')
        print(f'|   [StatsPrint]')
        print(f'|   |-- Name: {c.trainer.name}')
        print(f'|   |-- Epoch: {c.epoch}')
        print(f'|   |-- Total sample count: {c.sample_count}')
        print(f'|   |-- Total batch count: {c.iteration}')
        print(f'|   |-- Learning rates: {r(current_lrs)}')
        print(f'|   |-- Train')
        print(f'|   |     |-- loss{gdim(t_loss_monitor_result)}')
        print(f'|   |     |     |-- {self._get_print_from_monitor_result(t_loss_monitor_result)}')
        print(f'|   |     |-- metrics')
        print(f'|   |           |-- {gmfp(train_metric_monitor_results, prefix="|   |           |-- ")}')
        print(f'|   |')
        print(f'|   |-- Validation')
        print(f'|         |-- loss{gdim(v_loss_monitor_result)}')
        print(f'|         |     |-- {self._get_print_from_monitor_result(v_loss_monitor_result)}')
        print(f'|         |-- metrics')
        print(f'|               |-- {gmfp(val_metric_monitor_results, prefix="|               |-- ")}')
        print('------------------------------------------------------')
        print('') #EMPTY LINE SEPARATOR
