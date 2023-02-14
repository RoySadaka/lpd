from lpd.enums import Phase, State, MonitorType, MonitorMode, StatsType
from lpd.callbacks.callback_base import CallbackBase
from lpd.callbacks.callback_context import CallbackContext
from lpd.callbacks.callback_monitor import CallbackMonitor, CallbackMonitorResult
from lpd.metrics.metric_base import MetricConfusionMatrixBase
from lpd.utils.torch_utils import get_lrs_from_optimizer
from typing import Union, List, Optional, Dict, Iterable
from lpd.enums.confusion_matrix_based_metric import ConfusionMatrixBasedMetric as metric


class StatsPrint(CallbackBase):
    GREEN_PRINT_COLOR = "\033[92m"
    CYAN_PRINT_COLOR = "\033[96m"
    LIGHT_YELLOW_PRINT_COLOR = "\u001b[38;5;222m"
    BOLD_PRINT_COLOR = "\033[1m"
    END_PRINT_COLOR = "\033[0m"

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

            train_best_confusion_matrix_monitor - Provide if you wish also to track the best confusion matrix according to the metric defined in the monitor 
                                                  If None, best confusion matrix will not be tracked.
                                                  val_best_confusion_matrix_monitor will be applied automatically based on train_best_confusion_matrix_monitor

    """

    def __init__(self, apply_on_phase: Phase=Phase.EPOCH_END, 
                       apply_on_states: Union[State, List[State]]=State.EXTERNAL, 
                       round_values_on_print_to: Optional[int]=None,
                       train_metrics_monitors: Union[CallbackMonitor,Iterable[CallbackMonitor]]=None,
                       print_confusion_matrix: bool=False,
                       print_confusion_matrix_normalized: bool=False,
                       train_best_confusion_matrix_monitor: CallbackMonitor=None):
        super(StatsPrint, self).__init__(apply_on_phase, apply_on_states, round_values_on_print_to)
        self.train_loss_monitor = CallbackMonitor(MonitorType.LOSS, StatsType.TRAIN, MonitorMode.MIN)
        self.val_loss_monitor = CallbackMonitor(MonitorType.LOSS, StatsType.VAL, MonitorMode.MIN)
        self.train_metrics_monitors = self._parse_train_metrics_monitors(train_metrics_monitors)
        self.print_confusion_matrix = print_confusion_matrix
        self.print_confusion_matrix_normalized = print_confusion_matrix_normalized
        self.stats_type_to_best_confusion_matrix_monitor: Dict[StatsType, CallbackMonitor] = self._get_stats_type_to_best_confusion_matrix_monitor(train_best_confusion_matrix_monitor)
        self.stats_type_to_best_confusion_matrix_text: Dict[StatsType,str] = {StatsType.TRAIN:'', StatsType.VAL:''}
        self.val_metric_monitors: List[CallbackMonitor] = None
        self._validate_stats_type()


    def _is_confusion_matrix_on(self):
        return self.print_confusion_matrix or self.print_confusion_matrix_normalized

    def _is_confusion_matrix_track_best_on(self):
        return self._is_confusion_matrix_on() and self.stats_type_to_best_confusion_matrix_monitor is not None

    def _get_stats_type_to_best_confusion_matrix_monitor(self, train_best_confusion_matrix_monitor: CallbackMonitor):
        if not self._is_confusion_matrix_on():
            return None

        if train_best_confusion_matrix_monitor is None:
            return None

        assert train_best_confusion_matrix_monitor.stats_type == StatsType.TRAIN, f'train_best_confusion_matrix_monitor must be with stats_type={StatsType.TRAIN}'

        return {StatsType.TRAIN:train_best_confusion_matrix_monitor,
                StatsType.VAL: CallbackMonitor(monitor_type=train_best_confusion_matrix_monitor.monitor_type, 
                                               stats_type=StatsType.VAL,
                                               monitor_mode=train_best_confusion_matrix_monitor.monitor_mode,
                                               metric_name=train_best_confusion_matrix_monitor.metric_name,
                                               threshold_checker=train_best_confusion_matrix_monitor.threshold_checker)}

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

        trainer_metrics = callback_context.trainer.metrics
        if self.train_metrics_monitors is None:
            self.train_metrics_monitors = [CallbackMonitor(MonitorType.METRIC, StatsType.TRAIN, MonitorMode.MAX, metric_name=trainer_metric.name) for trainer_metric in trainer_metrics]

        self.val_metric_monitors = []
        for m in self.train_metrics_monitors:
            self.val_metric_monitors.append(CallbackMonitor(m.monitor_type, StatsType.VAL, m.monitor_mode, patience=m.patience, metric_name=m.metric_name))

    def _get_print_from_monitor_result_aux(self, value):
        if len(value.shape) == 0:
            return value.item()
        return value.tolist()

    def _get_print_from_monitor_result(self, monitor_result: CallbackMonitorResult) -> str:
        r = self.round_to #READABILITY
        aux = self._get_print_from_monitor_result_aux
        mtr = monitor_result #READABILITY
        return f'curr:{r(aux(r(mtr.new_value)))}, prev:{r(aux(r(mtr.prev_value)))}, best:{r(aux(r(mtr.new_best)))}, change_from_prev:{r(aux(r(mtr.change_from_previous)))}, change_from_best:{r(aux(r(mtr.change_from_best)))}'

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
            return StatsPrint.GREEN_PRINT_COLOR + ' IMPROVED' + StatsPrint.END_PRINT_COLOR
        return ''

    def _get_print_confusion_matrix(self, state: State, callback_context: CallbackContext, prefix: str='') -> str:
        if not self._is_confusion_matrix_on():
            if state == State.TRAIN:
                return '┃   ┃'
            elif state == State.VAL:
                return '┃'

        c = callback_context #READABILITY

        if state == State.TRAIN:
            total_end_delimiter = '\n┃   ┃           ┗' + '━' * 84
            row_gap = '┃   ┃           ┃\n'
            row_start = '┃   ┃           ┣━━ '
            stats = c.train_stats
        elif state == State.VAL:
            total_end_delimiter = '\n┃               ┗' + '━' * 84
            row_gap = '┃               ┃\n'
            row_start = '┃               ┣━━ '
            stats = c.val_stats
        
        cm = stats.confusion_matrix
        assert cm is not None, '[StatsPrint] - print_confusion_matrix is set to True, but no confusion matrix based metric was set on trainer'
        
        current_cm_str = cm.confusion_matrix_string(normalized = self.print_confusion_matrix_normalized)
        current_cm_str_split = current_cm_str.split('\n')

        if self._is_confusion_matrix_track_best_on():
            if state == State.TRAIN:
                stats_type = StatsType.TRAIN
                cm = c.train_stats.confusion_matrix
            elif state == State.VAL:
                stats_type = StatsType.VAL
                cm = c.val_stats.confusion_matrix

            callback_monitor_result = self.stats_type_to_best_confusion_matrix_monitor[stats_type].track(c)
            improved_colored_txt = self._get_did_improved_colored(callback_monitor_result)
            if callback_monitor_result.has_improved():
                self.stats_type_to_best_confusion_matrix_text[stats_type] = cm.confusion_matrix_string(normalized = self.print_confusion_matrix_normalized)

            best_cm_str = self.stats_type_to_best_confusion_matrix_text[stats_type]
            
            best_cm_str_split = best_cm_str.split('\n')
            assert len(current_cm_str_split) == len(best_cm_str_split)

            max_curr_line_length = max(len(line) for line in current_cm_str_split)
            lines = []
            for idx,(curr,best) in enumerate(zip(current_cm_str_split, best_cm_str_split)):
                if idx == 0: # HEADER
                    lines.append(f'{curr}{improved_colored_txt}  ┳  {StatsPrint.BOLD_PRINT_COLOR}Best {StatsPrint.END_PRINT_COLOR}{best}')
                else:
                    spaces = ' '*(max_curr_line_length+min(9,max(0, len(improved_colored_txt)))-len(curr)) # BUT HACKY " IMPROVED" IS AT MOST 9 LETTERS, BUT HAS COLORING
                    lines.append(f'{prefix}{curr}{spaces}  ┃  {best}')
            cm_str = '\n'.join(lines)
        else:
            cm_str = '\n'.join([f'{prefix}{x}' if idx > 0 else x for idx,x in enumerate(current_cm_str_split)])

        return f'{row_gap}{row_start}{cm_str}{total_end_delimiter}'

    def __call__(self, callback_context: CallbackContext):
        c = callback_context #READABILITY

        self._ensure_metrics_created(c)

        # INVOKE MONITORS
        train_loss_monitor_result = self.train_loss_monitor.track(c)
        val_loss_monitor_result = self.val_loss_monitor.track(c)

        train_metric_monitor_results = [m.track(c) for m in self.train_metrics_monitors]
        val_metric_monitor_results = [m.track(c) for m in self.val_metric_monitors]

        current_lrs = get_lrs_from_optimizer(c.trainer.optimizer)

        rt = self.round_to #READABILITY
        gdic = self._get_did_improved_colored #READABILITY 
        gpfm = self._get_print_from_metrics #READABILITY 
        gpcm = self._get_print_confusion_matrix #READABILITY 

        Y,C,E = StatsPrint.LIGHT_YELLOW_PRINT_COLOR, StatsPrint.CYAN_PRINT_COLOR, StatsPrint.END_PRINT_COLOR 
        print(f'┏{"━"*100}')
        print(f'┃   [StatsPrint]')
        print(f'┃   ┏━━ {Y}Name{E}: "{c.trainer.name}"')
        print(f'┃   ┣━━ {Y}Epoch{E}: {c.epoch}')
        print(f'┃   ┣━━ {Y}Total sample count{E}: {c.sample_count}')
        print(f'┃   ┣━━ {Y}Total batch count{E}: {c.iteration}')
        print(f'┃   ┣━━ {Y}Learning rates{E}: {rt(current_lrs)}')
        print(f'┃   ┣━━ {C}Train{E}')
        print(f'┃   ┃     ┣━━ {Y}loss{E}{gdic(train_loss_monitor_result)}')
        print(f'┃   ┃     ┃     ┗━━ {self._get_print_from_monitor_result(train_loss_monitor_result)}')
        print(f'┃   ┃     ┗━━ {Y}metrics{E}')
        print(f'┃   ┃           ┣━━ {gpfm(train_metric_monitor_results, prefix="┃   ┃           ┣━━ ")}')
        print(gpcm(State.TRAIN, c, prefix="┃   ┃           ┃   "))
        print(f'┃   ┃')
        print(f'┃   ┗━━ {C}Validation{E}')
        print(f'┃         ┣━━ {Y}loss{E}{gdic(val_loss_monitor_result)}')
        print(f'┃         ┃     ┗━━ {self._get_print_from_monitor_result(val_loss_monitor_result)}')
        print(f'┃         ┗━━ {Y}metrics{E}')
        print(f'┃               ┣━━ {gpfm(val_metric_monitor_results, prefix="┃               ┣━━ ")}')
        print(gpcm(State.VAL, c, prefix="┃               ┃   "))
        print(f'┗{"━"*100}')
        print('') #EMPTY LINE SEPARATOR
