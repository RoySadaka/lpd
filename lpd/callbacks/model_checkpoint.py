from lpd.enums import Phase, State, MonitorType, MonitorMode, StatsType
from lpd.callbacks.callback_base import CallbackBase
from lpd.callbacks.callback_context import CallbackContext
from lpd.callbacks.callback_monitor import CallbackMonitor, CallbackMonitorResult
import lpd.utils.file_utils as fu
from lpd.utils.torch_utils import save_checkpoint
from typing import Union, List, Optional, Dict

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

    def __init__(self,  cb_phase: Phase=Phase.EPOCH_END, 
                        apply_on_states: Union[State, List[State]]=State.EXTERNAL,
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
