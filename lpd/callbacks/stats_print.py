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

            print_confusion_matrix - If one of the metrics is of type MetricConfusionMatrixBase, setting this to True will print the confusion matrix
                                     If False, confusion matrix will not be printed (but the metric stats will remain)
            print_confusion_matrix_normalized - same as 'print_confusion_matrix', except it prints the normalized confusion matrix
    """

    def __init__(self, apply_on_phase: Phase=Phase.EPOCH_END, 
                       apply_on_states: Union[State, List[State]]=State.EXTERNAL, 
                       round_values_on_print_to: Optional[int]=None,
                       train_metrics_monitors: Union[CallbackMonitor,Iterable[CallbackMonitor]]=None,
                       print_confusion_matrix: bool=False,
                       print_confusion_matrix_normalized: bool=False):
        super(StatsPrint, self).__init__(apply_on_phase, apply_on_states, round_values_on_print_to)
        self.train_loss_monitor = CallbackMonitor(MonitorType.LOSS, StatsType.TRAIN, MonitorMode.MIN)
        self.val_loss_monitor = CallbackMonitor(MonitorType.LOSS, StatsType.VAL, MonitorMode.MIN)
        self.train_metrics_monitors = self._parse_train_metrics_monitors(train_metrics_monitors)
        self.print_confusion_matrix = print_confusion_matrix
        self.print_confusion_matrix_normalized = print_confusion_matrix_normalized
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
                    raise ValueError(f'[StatsPrint] - train_metrics_monitors contains monitor with stats_type {m.stats_type}, expected {StatsType.TRAIN}')
            
    def _ensure_metrics_created(self, callback_context: CallbackContext):
        if self.train_metrics_monitors is not None and self.val_metric_monitors is not None:
            # ENSURED ALREADY
            return

        metric_names = callback_context.trainer.metric_name_to_func.keys()
        if self.train_metrics_monitors is None:
            self.train_metrics_monitors = [CallbackMonitor(MonitorType.METRIC, StatsType.TRAIN, MonitorMode.MAX, metric_name=metric_name) for metric_name in metric_names]

        self.val_metric_monitors = []
        for m in self.train_metrics_monitors:
            self.val_metric_monitors.append(CallbackMonitor(m.monitor_type, StatsType.VAL, m.monitor_mode, patience=m.patience, metric_name=m.metric_name))

    def _get_print_from_monitor_result(self, monitor_result: CallbackMonitorResult) -> str:
        r = self.round_to #READABILITY
        mtr = monitor_result #READABILITY
        return f'curr:{r(mtr.new_value)}, prev:{r(mtr.prev_value)}, best:{r(mtr.new_best)}, change_from_prev:{r(mtr.change_from_previous)}, change_from_best:{r(mtr.change_from_best)}'

    def _get_print_from_metrics(self, train_metric_monitor_results: Iterable[CallbackMonitorResult], prefix: str='') -> str:
        gdic = self._get_did_improved_colored #READABILITY 

        if len(train_metric_monitor_results) == 0:
            return 'no metrics found'
        prints = []
        pre = ''
        for monitor_result in train_metric_monitor_results:
            name = monitor_result.name
            prints.append(f'{pre}name: "{name}"{gdic(monitor_result)}, {self._get_print_from_monitor_result(monitor_result)}')
            pre = prefix # APPEND PREFIX FROM THE SECOND METRIC AND ON

        return '\n'.join(prints)

    def _get_did_improved_colored(self, monitor_result):
        if monitor_result.has_improved():
            return self.GREEN_PRINT_COLOR + ' IMPROVED' + self.END_PRINT_COLOR
        return ''

    def _get_print_confusion_matrix(self, state: State, callback_context: CallbackContext, prefix: str='') -> str:
        if not self.print_confusion_matrix and not self.print_confusion_matrix_normalized:
            if state == State.TRAIN:
                return '|   |'
            elif state == State.VAL:
                return '|'

        if state == State.TRAIN:
            total_end_delimeter = '\n|   |           |' + '_' * 20
            row_start = '|   |           |-- '
            stats = callback_context.train_stats
        elif state == State.VAL:
            total_end_delimeter = '\n|               |' + '_' * 20
            row_start = '|               |-- '
            stats = callback_context.val_stats
        
        cm = stats.confusion_matrix
        if cm is None:
            raise ValueError('[StatsPrint] - print_confusion_matrix is set to True, but no confusion matrix based metric was set on trainer')
        return f'{row_start}{cm.confusion_matrix_string(prefix, normalized = self.print_confusion_matrix_normalized)}{total_end_delimeter}'
        

    def __call__(self, callback_context: CallbackContext):
        c = callback_context #READABILITY 
        r = self.round_to #READABILITY
        gdic = self._get_did_improved_colored #READABILITY 
        gpfm = self._get_print_from_metrics #READABILITY 
        gpcm = self._get_print_confusion_matrix #READABILITY 

        self._ensure_metrics_created(c)

        # INVOKE MONITORS
        t_loss_monitor_result = self.train_loss_monitor.track(c)
        v_loss_monitor_result = self.val_loss_monitor.track(c)

        train_metric_monitor_results = [m.track(c) for m in self.train_metrics_monitors]
        val_metric_monitor_results = [m.track(c) for m in self.val_metric_monitors]

        current_lrs = get_lrs_from_optimizer(c.trainer.optimizer)

        print('------------------------------------------------------')
        print(f'|   [StatsPrint]')
        print(f'|   |-- Name: "{c.trainer.name}"')
        print(f'|   |-- Epoch: {c.epoch}')
        print(f'|   |-- Total sample count: {c.sample_count}')
        print(f'|   |-- Total batch count: {c.iteration}')
        print(f'|   |-- Learning rates: {r(current_lrs)}')
        print(f'|   |-- Train')
        print(f'|   |     |-- loss{gdic(t_loss_monitor_result)}')
        print(f'|   |     |     |-- {self._get_print_from_monitor_result(t_loss_monitor_result)}')
        print(f'|   |     |-- metrics')
        print(f'|   |           |-- {gpfm(train_metric_monitor_results, prefix="|   |           |-- ")}')
        print(gpcm(State.TRAIN, c, prefix="\n|   |           |   "))
        print(f'|   |')
        print(f'|   |-- Validation')
        print(f'|         |-- loss{gdic(v_loss_monitor_result)}')
        print(f'|         |     |-- {self._get_print_from_monitor_result(v_loss_monitor_result)}')
        print(f'|         |-- metrics')
        print(f'|               |-- {gpfm(val_metric_monitor_results, prefix="|               |-- ")}')
        print(gpcm(State.VAL, c, prefix="\n|               |   "))
        print('------------------------------------------------------')
        print('') #EMPTY LINE SEPARATOR
