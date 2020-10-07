from math import inf
from typing import Union, List, Optional, Dict
from lpd.utils.torch_utils import save_checkpoint
from torch.utils.tensorboard import SummaryWriter
from lpd.enums import CallbackPhase, TrainerState, MonitorType, MonitorMode, StatsType
import lpd.utils.file_utils as fu
from lpd.trainer_stats import TrainerStats

class CallbackContext():
    #REPRESENTS THE INPUT TO THE CALLBACK, NOTICE, SOME VALUES MIGHT BE NONE, DEPENDING ON THE PHASE OF THE CALLBACK
    def __init__(self, trainer):
        self.epoch = trainer._current_epoch
        self.train_stats = trainer.train_stats
        self.val_stats = trainer.val_stats
        self.test_stats = trainer.test_stats
        self.trainer_state = trainer.state
        self.trainer_phase = trainer.phase
        self.trainer = trainer

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
        self.previous = self.maximum if monitor_mode == MonitorMode.MIN else self.minimum
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

class CallbackBase():
    """
        Agrs:
            cb_phase - (lpd.enums.CallbackPhase) the phase to invoke this callback
            round_values_on_print_to - optional, it will round the numerical values in the prints
            apply_on_states - (lpd.enums.TrainerState) state or list of states to invoke this parameter (under the relevant cb_phase), None will invoke it on all states
    """

    def __init__(self, cb_phase: CallbackPhase, 
                       apply_on_states: Union[TrainerState, List[TrainerState]], 
                       round_values_on_print_to: Optional[int]=None):
        self.cb_phase = cb_phase
        if self.cb_phase is None:
            print('[CallbackBase][Error!] - No callback phase was provided')

        self.apply_on_states = apply_on_states
        self.round_values_on_print_to = round_values_on_print_to

    def round_to(self, value: int):
        if self.round_values_on_print_to:
            return round(value, self.round_values_on_print_to)
        return value

    def should_apply_on_phase(self, callback_context: CallbackContext):
        if isinstance(self.cb_phase, CallbackPhase):
            return callback_context.trainer_phase == self.cb_phase
        raise ValueError('[CallbackBase] - got bad value for cb_phase')

    def should_apply_on_state(self, callback_context: CallbackContext):
        if self.apply_on_states is None:
            return True

        if isinstance(self.apply_on_states, list):
            for state in self.apply_on_states:
                if isinstance(state, TrainerState):
                    if callback_context.trainer_state == state:
                        return True
            return False

        if isinstance(self.apply_on_states, TrainerState):
            return callback_context.trainer_state == self.apply_on_states

        raise ValueError('[CallbackBase] - got bad value for apply_on_states')

class SchedulerStep(CallbackBase):
    """This callback will invoke a "step()" on the scheduler.

        Agrs:
            scheduler_parameters_func - Since some schedulers takes parameters in step(param1, param2...)
                And other schedulers step() are parameterless, provide:
                a function (or lambda) that except trainer and returns whatever information needed,
                e.g. for scheduler that takes val_loss as parameter, initialize like this:
                    SchedulerStep(scheduler_parameters_func=lambda trainer: trainer.val_stats.get_loss())
                if your scheduler step does not expect parameters, leave scheduler_parameters_func = None
            cb_phase - see in CallbackBase
            apply_on_states - see in CallbackBase
    """
    def __init__(self, cb_phase: CallbackPhase=CallbackPhase.EPOCH_END, 
                       apply_on_states: Union[TrainerState, List[TrainerState]]=TrainerState.EXTERNAL,
                       scheduler_parameters_func=None):
        super(SchedulerStep, self).__init__(cb_phase=cb_phase, apply_on_states=apply_on_states)
        self.scheduler_parameters_func = scheduler_parameters_func

    def __call__(self, callback_context):
        if callback_context.trainer.scheduler is None:
            print('[SchedulerStep] - no scheduler defined in trainer')
            return
        if self.scheduler_parameters_func:
            callback_context.trainer.scheduler.step(self.scheduler_parameters_func(callback_context.trainer))
        else:
            callback_context.trainer.scheduler.step()

class StatsPrint(CallbackBase):
    """
        Informative summary of the trainer state, most likely at the end of the epoch, 
        but you can change cb_phase and apply_on_states if you need it on a different phases
        Args:
            cb_phase - see in CallbackBase
            apply_on_states - see in CallbackBase
            round_values_on_print_to - see in CallbackBase
    """

    def __init__(self, cb_phase: CallbackPhase=CallbackPhase.EPOCH_END, 
                       apply_on_states: Union[TrainerState, List[TrainerState]]=TrainerState.EXTERNAL, 
                       round_values_on_print_to=None,
                       metric_names=None):
        super(StatsPrint, self).__init__(cb_phase, apply_on_states, round_values_on_print_to)
        self.metric_names = metric_names or set()
        self.train_loss_monitor = CallbackMonitor(None, MonitorType.LOSS, StatsType.TRAIN, MonitorMode.MIN)
        self.val_loss_monitor = CallbackMonitor(None, MonitorType.LOSS, StatsType.VAL, MonitorMode.MIN)

        self.train_metric_name_to_monitor = {}
        self.val_metric_name_to_monitor = {}
        for metric_name in self.metric_names:
            self.train_metric_name_to_monitor[metric_name] = CallbackMonitor(None, MonitorType.METRIC, StatsType.TRAIN, MonitorMode.MAX, metric_name)
            self.val_metric_name_to_monitor[metric_name] = CallbackMonitor(None, MonitorType.METRIC, StatsType.VAL, MonitorMode.MAX, metric_name)

        self.GREEN_PRINT_COLOR = "\033[92m"
        self.END_PRINT_COLOR = "\033[0m"

    def _get_current_lr(self, optimizer):
        #CURRENTLY WILL RETURN ONLY FOR param_groups[0]
        return optimizer.param_groups[0]['lr']

    def _get_print_from_monitor_result(self, monitor_result: CallbackMonitorResult) -> str:
        r = self.round_to #READABILITY
        mtr = monitor_result #READABILITY
        return f'curr:{r(mtr.new_value)}, prev:{r(mtr.prev_value)}, best:{r(mtr.new_best)}, change_from_prev:{r(mtr.change_from_previous)}, change_from_best:{r(mtr.change_from_best)}'

    def _get_print_from_metrics(self, metric_name_to_monitor_result: Dict[str, CallbackMonitorResult]) -> str:
        gdim = self._get_did_improved_colored #READABILITY 

        if len(metric_name_to_monitor_result) == 0:
            return 'no metrics found'
        prints = []
        for metric_name,monitor_result in metric_name_to_monitor_result.items():
            prints.append(f'name: {metric_name} {gdim(monitor_result)}, {self._get_print_from_monitor_result(monitor_result)}')

        return '\n'.join(prints)

    def _get_did_improved_colored(self, monitor_result):
        if monitor_result.has_improved():
            return self.GREEN_PRINT_COLOR + 'IMPROVED' + self.END_PRINT_COLOR
        return ''


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

        current_lr = self._get_current_lr(c.trainer.optimizer)

        print('------------------------------------------------------')
        print(f'|   [StatsPrint]')
        print(f'|   |-- Name: {c.trainer.name}')
        print(f'|   |-- Epoch: {c.epoch}')
        print(f'|   |-- Learning rate: {r(current_lr)}')
        print(f'|   |-- Train')
        print(f'|   |     |-- loss {gdim(t_loss_monitor_result)}')
        print(f'|   |     |     |-- {self._get_print_from_monitor_result(t_loss_monitor_result)}')
        print(f'|   |     |-- metrics')
        print(f'|   |           |-- {gmfp(train_metric_name_to_monitor_result)}')
        print(f'|   |')
        print(f'|   |-- Validation')
        print(f'|         |-- loss {gdim(v_loss_monitor_result)}')
        print(f'|         |     |-- {self._get_print_from_monitor_result(v_loss_monitor_result)}')
        print(f'|         |-- metrics')
        print(f'|               |-- {gmfp(val_metric_name_to_monitor_result)}')
        print('------------------------------------------------------')
        print('') #EMPTY LINE SEPARATOR

class ModelCheckPoint(CallbackBase):
    """
        Saving a checkpoint when a monitored loss has improved.
        Checkpoint will save the model, optimizer, scheduler and epoch number
        Args:
            cb_phase - see in CallbackBase
            apply_on_states - see in CallbackBase
            checkpoint_dir - the folder to dave the model, if None was passed, will use current folder
            checkpoint_file_name - the name of the file that will be saved
            monitor_type - e.g. lpd.enums.MonitorType.LOSS, what to monitor (see CallbackMonitor)
            stats_type - e.g. lpd.enums.StatsType.VAL (see CallbackMonitor)
            monitor_mode - e.g. lpd.enums.MonitorMode.MIN (see CallbackMonitor)
            metric_name - name if lpd.enums.MonitorType.METRIC
            save_best_only - if True, will override previous best model, else, will keep both
            verbose - 0=no print, 1=print
            round_values_on_print_to - see in CallbackBase
    """

    def __init__(self,  cb_phase: CallbackPhase=CallbackPhase.EPOCH_END, 
                        apply_on_states: Union[TrainerState, List[TrainerState]]=TrainerState.EXTERNAL,
                        checkpoint_dir: str=None, 
                        checkpoint_file_name: str='checkpoint', 
                        monitor_type: MonitorType=MonitorType.LOSS, 
                        stats_type: StatsType=StatsType.VAL, 
                        monitor_mode: MonitorMode=MonitorMode.MIN, 
                        metric_name: str=None,
                        save_best_only: bool=False, 
                        verbose: int=1,
                        round_values_on_print_to: int=None):
        super(ModelCheckPoint, self).__init__(cb_phase, apply_on_states, round_values_on_print_to)
        self.checkpoint_dir = checkpoint_dir
        if self.checkpoint_dir is None:
            raise ValueError("[ModelCheckPoint] - checkpoint_dir was not provided")
        self.checkpoint_file_name = checkpoint_file_name
        self.monitor = CallbackMonitor(None, monitor_type, stats_type, monitor_mode, metric_name)
        self.save_best_only = save_best_only
        self.verbose = verbose  # VERBOSITY MODE, 0 OR 1.
        self._ensure_folder_created()

    def _ensure_folder_created(self):
        if not fu.is_folder_exists(self.checkpoint_dir):
            fu.create_folder(self.checkpoint_dir)

    def __call__(self, callback_context: CallbackContext):
        c = callback_context #READABILITY DOWN THE ROAD
        r = self.round_to #READABILITY DOWN THE ROAD
        
        monitor_result = self.monitor.track(callback_context)
        if monitor_result.has_improved():
            msg = f'[ModelCheckPoint] - {monitor_result.description} improved from {r(monitor_result.prev_best)} to {r(monitor_result.new_best)}'
            if self.save_best_only:
                full_path = f'{self.checkpoint_dir}{self.checkpoint_file_name}_best_only'
            else:
                full_path = f'{self.checkpoint_dir}{self.checkpoint_file_name}_epoch_{c.epoch}'
            save_checkpoint(full_path, c.epoch, c.trainer.model, c.trainer.optimizer, c.trainer.scheduler, msg=msg, verbose=self.verbose)
        else:
            if self.verbose:
                print(f'[ModelCheckPoint] - {monitor_result.description} did not improved from {monitor_result.prev_best}.')

class Tensorboard(CallbackBase):
    """ 
        Writes entries directly to event files in the summary_writer_dir to be
        consumed by TensorBoard.
        Args:
            cb_phase - see in CallbackBase
            apply_on_states - see in CallbackBase
            summary_writer_dir - the folder path to save tensorboard output
                                 if passed None, will write to the current dir
    """

    def __init__(self, cb_phase: CallbackPhase=CallbackPhase.EPOCH_END, 
                        apply_on_states: Union[TrainerState, List[TrainerState]]=TrainerState.EXTERNAL,
                        summary_writer_dir: str=None):
        super(Tensorboard, self).__init__(cb_phase, apply_on_states)
        self.TRAIN_NAME = 'Train'
        self.VAL_NAME = 'Val'
        self.summary_writer_dir = summary_writer_dir
        if self.summary_writer_dir is None:
            raise ValueError("[Tensorboard] - summary_writer_dir was not provided")
        self.tensorboard_writer = SummaryWriter(summary_writer_dir + 'tensorboard_files')

    def _write_to_summary(self, phase_name: str ,epoch: int, stats: TrainerStats):
        self.tensorboard_writer.add_scalar(f'{phase_name} loss', stats.get_loss(), global_step=epoch)
        for metric_name, value in stats.get_metrics().items():
            self.tensorboard_writer.add_scalar(metric_name, value, global_step=epoch)

    def __call__(self, callback_context: CallbackContext):
        c = callback_context #READABILITY DOWN THE ROAD
        self._write_to_summary(self.TRAIN_NAME, c.epoch, c.train_stats)
        self._write_to_summary(self.VAL_NAME, c.epoch, c.val_stats)

class EarlyStopping(CallbackBase):
    """
        Stop training when a monitored loss has stopped improving.
        Args:
            cb_phase - see in CallbackBase
            apply_on_states - see in CallbackBase
            patience - int or None (will be set to inf) track how many epochs/iterations without improvements in monitoring
            monitor_type - e.g. lpd.enums.MonitorType.LOSS, what to monitor (see CallbackMonitor)
            stats_type - e.g. lpd.enums.StatsType.VAL (see CallbackMonitor)
            monitor_mode - e.g. lpd.enums.MonitorMode.MIN (see CallbackMonitor)
            metric_name - name if lpd.enums.MonitorType.METRIC is being monitored
            verbose - 0 = no print, 1 = print all, 2 = print save only
    """

    def __init__(self, 
                    cb_phase: CallbackPhase=CallbackPhase.EPOCH_END, 
                    apply_on_states: Union[TrainerState, List[TrainerState]]=TrainerState.EXTERNAL,
                    patience: int=0, 
                    monitor_type: MonitorType=MonitorType.LOSS, 
                    stats_type: StatsType=StatsType.VAL, 
                    monitor_mode: MonitorMode=MonitorMode.MIN, 
                    metric_name: Optional[str]=None,
                    verbose=1):
        super(EarlyStopping, self).__init__(cb_phase, apply_on_states)
        self.monitor = CallbackMonitor(patience, monitor_type, stats_type, monitor_mode, metric_name)
        self.verbose = verbose

    def __call__(self, callback_context: CallbackContext):
        c = callback_context #READABILITY DOWN THE ROAD

        monitor_result = self.monitor.track(c)

        if monitor_result.has_patience() and self.verbose:
            print(f'[EarlyStopping] - patience:{monitor_result.patience_left} epochs')
        
        if not monitor_result.has_patience():
            c.trainer.stop_training()
            if self.verbose > 0:
                print(f'[EarlyStopping] - stopping on epoch {c.epoch}')


