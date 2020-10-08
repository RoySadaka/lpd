from lpd.trainer_stats import TrainerStats
from lpd.enums import Phase, State, MonitorType, MonitorMode, StatsType
from lpd.callbacks.callback_base import CallbackBase
from lpd.callbacks.callback_context import CallbackContext
from lpd.callbacks.callback_monitor import CallbackMonitorResult

from torch.utils.tensorboard import SummaryWriter
from typing import Union, List, Optional, Dict

class Tensorboard(CallbackBase):
    """ 
        Writes entries directly to event files in the summary_writer_dir to be
        consumed by TensorBoard.
        Args:
            apply_on_phase - see in CallbackBase
            apply_on_states - see in CallbackBase
            summary_writer_dir - the folder path to save tensorboard output
                                 if passed None, will write to the current dir
    """

    def __init__(self, apply_on_phase: Phase=Phase.EPOCH_END, 
                        apply_on_states: Union[State, List[State]]=State.EXTERNAL,
                        summary_writer_dir: str=None):
        super(Tensorboard, self).__init__(apply_on_phase, apply_on_states)
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
