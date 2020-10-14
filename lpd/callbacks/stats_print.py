from lpd.enums import Phase, State, MonitorType, MonitorMode, StatsType
from lpd.callbacks.callback_base import CallbackBase
from lpd.callbacks.callback_context import CallbackContext
from lpd.callbacks.callback_monitor import CallbackMonitor, CallbackMonitorResult
from lpd.utils.torch_utils import get_lrs_from_optimizer
from typing import Union, List, Optional, Dict, Iterable

class StatsPrint(CallbackBase):
    """
        Informative summary of the trainer state, most likely at the end of the epoch, 
        but you can change apply_on_phase and apply_on_states if you need it on a different phases
        Args:
            apply_on_phase - see in CallbackBase
            apply_on_states - see in CallbackBase
            round_values_on_print_to - see in CallbackBase
    """

    def __init__(self, apply_on_phase: Phase=Phase.EPOCH_END, 
                       apply_on_states: Union[State, List[State]]=State.EXTERNAL, 
                       round_values_on_print_to: Optional[int]=None,
                       metric_names: Optional[Union[str,Iterable]]=None):
        super(StatsPrint, self).__init__(apply_on_phase, apply_on_states, round_values_on_print_to)
        self.metric_names = self._extract_metric_names(metric_names)
        self.train_loss_monitor = CallbackMonitor(None, MonitorType.LOSS, StatsType.TRAIN, MonitorMode.MIN)
        self.val_loss_monitor = CallbackMonitor(None, MonitorType.LOSS, StatsType.VAL, MonitorMode.MIN)

        self.train_metric_name_to_monitor = {}
        self.val_metric_name_to_monitor = {}
        for metric_name in self.metric_names:
            self.train_metric_name_to_monitor[metric_name] = CallbackMonitor(None, MonitorType.METRIC, StatsType.TRAIN, MonitorMode.MAX, metric_name)
            self.val_metric_name_to_monitor[metric_name] = CallbackMonitor(None, MonitorType.METRIC, StatsType.VAL, MonitorMode.MAX, metric_name)

        self.GREEN_PRINT_COLOR = "\033[92m"
        self.END_PRINT_COLOR = "\033[0m"

    def _extract_metric_names(self, metric_names):
        result = set()
        if isinstance(metric_names, str):
            result.add(metric_names)
        elif isinstance(metric_names, Iterable):
            result = set(metric_names)
        return result

    def _get_print_from_monitor_result(self, monitor_result: CallbackMonitorResult) -> str:
        r = self.round_to #READABILITY
        mtr = monitor_result #READABILITY
        return f'curr:{r(mtr.new_value)}, prev:{r(mtr.prev_value)}, best:{r(mtr.new_best)}, change_from_prev:{r(mtr.change_from_previous)}, change_from_best:{r(mtr.change_from_best)}'

    def _get_print_from_metrics(self, metric_name_to_monitor_result: Dict[str, CallbackMonitorResult], prefix: str='') -> str:
        gdim = self._get_did_improved_colored #READABILITY 

        if len(metric_name_to_monitor_result) == 0:
            return 'no metrics found'
        prints = []
        pre = ''
        for metric_name,monitor_result in metric_name_to_monitor_result.items():
            prints.append(f'{pre}name: {metric_name}{gdim(monitor_result)}, {self._get_print_from_monitor_result(monitor_result)}')
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

        # INVOKE MONITORS
        t_loss_monitor_result = self.train_loss_monitor.track(c)
        v_loss_monitor_result = self.val_loss_monitor.track(c)
        train_metric_name_to_monitor_result = {}
        val_metric_name_to_monitor_result = {}
        for metric_name in self.metric_names:
            train_metric_name_to_monitor_result[metric_name] = self.train_metric_name_to_monitor[metric_name].track(c)
            val_metric_name_to_monitor_result[metric_name]   = self.val_metric_name_to_monitor[metric_name].track(c)

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
        print(f'|   |           |-- {gmfp(train_metric_name_to_monitor_result, prefix="|   |           |-- ")}')
        print(f'|   |')
        print(f'|   |-- Validation')
        print(f'|         |-- loss{gdim(v_loss_monitor_result)}')
        print(f'|         |     |-- {self._get_print_from_monitor_result(v_loss_monitor_result)}')
        print(f'|         |-- metrics')
        print(f'|               |-- {gmfp(val_metric_name_to_monitor_result, prefix="|               |-- ")}')
        print('------------------------------------------------------')
        print('') #EMPTY LINE SEPARATOR
