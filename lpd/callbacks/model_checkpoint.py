from lpd.enums import Phase, State, MonitorType, MonitorMode, StatsType
from lpd.callbacks.callback_base import CallbackBase
from lpd.callbacks.callback_context import CallbackContext
from lpd.callbacks.callback_monitor import CallbackMonitor, CallbackMonitorResult
import lpd.utils.file_utils as fu
from lpd.utils.torch_utils import save_checkpoint
from typing import Union, List, Optional, Dict

class ModelCheckPoint(CallbackBase):
    """
        Saving a checkpoint when a monitored loss/metric has improved.
        Checkpoint will save the model, optimizer, scheduler and epoch number.
        You can also configure it to save Full Trainer.

        Args:
            apply_on_phase - see in CallbackBase
            apply_on_states - see in CallbackBase
            checkpoint_dir - the folder to dave the model, if None was passed, will use current folder
            checkpoint_file_name - the name of the file that will be saved
            callback_monitor - to monitor the loss/metric for saving checkpoint when improved
            save_best_only - if True, will override previous best model, else, will keep both
            verbose - 0=no print, 1=print
            round_values_on_print_to - see in CallbackBase
            save_full_trainer - if True, will save all trainer parameters to be able to continue where you left off
    """

    def __init__(self,  apply_on_phase: Phase=Phase.EPOCH_END, 
                        apply_on_states: Union[State, List[State]]=State.EXTERNAL,
                        checkpoint_dir: str=None, 
                        checkpoint_file_name: str='checkpoint', 
                        callback_monitor: CallbackMonitor=None,
                        save_best_only: bool=False, 
                        verbose: int=1,
                        round_values_on_print_to: int=None,
                        save_full_trainer: bool=False):
        super(ModelCheckPoint, self).__init__(apply_on_phase, apply_on_states, round_values_on_print_to)
        if checkpoint_dir is None:
            raise ValueError("[ModelCheckPoint] - checkpoint_dir was not provided")
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file_name = checkpoint_file_name
        if callback_monitor is None:
            raise ValueError("[ModelCheckPoint] - callback_monitor was not provided")
        self.monitor = callback_monitor
        self.save_best_only = save_best_only
        self.verbose = verbose  # VERBOSITY MODE, 0 OR 1.
        self.save_full_trainer = save_full_trainer
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
                file_name = f'{self.checkpoint_file_name}_best_only'
            else:
                file_name = f'{self.checkpoint_file_name}_epoch_{c.epoch}'
            
            if self.save_full_trainer:
                c.trainer.save_trainer(self.checkpoint_dir, file_name, msg=msg, verbose=self.verbose)
            else:
                save_checkpoint(self.checkpoint_dir, file_name, c.trainer, msg=msg, verbose=self.verbose)
        else:
            if self.verbose:
                print(f'[ModelCheckPoint] - {monitor_result.description} did not improved from {monitor_result.prev_best}.')
