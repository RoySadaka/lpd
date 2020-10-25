from lpd.enums import Phase, State, MonitorType, MonitorMode, StatsType
from lpd.callbacks.callback_base import CallbackBase
from lpd.callbacks.callback_context import CallbackContext
from lpd.callbacks.callback_monitor import CallbackMonitor, CallbackMonitorResult
from typing import Union, List, Optional, Dict

class EarlyStopping(CallbackBase):
    """
        Stop training when a monitored loss/metric has stopped improving.
        Args:
            apply_on_phase - see in CallbackBase
            apply_on_states - see in CallbackBase
            callback_monitor - to monitor the loss/metric and stop training when stopped improving
            verbose - 0 = no print, 1 = print all, 2 = print save only
    """

    def __init__(self, 
                    apply_on_phase: Phase=Phase.EPOCH_END, 
                    apply_on_states: Union[State, List[State]]=State.EXTERNAL,
                    callback_monitor: CallbackMonitor=None,
                    verbose: int=1):
        super(EarlyStopping, self).__init__(apply_on_phase, apply_on_states)
        if callback_monitor is None:
            raise ValueError("[EarlyStopping] - callback_monitor was not provided")
        self.monitor = callback_monitor
        self.verbose = verbose

    def __call__(self, callback_context: CallbackContext):
        c = callback_context #READABILITY DOWN THE ROAD

        monitor_result = self.monitor.track(c)

        if monitor_result.has_patience() and self.verbose:
            print(f'[EarlyStopping] - patience:{monitor_result.patience_left} epochs')
        
        if not monitor_result.has_patience():
            c.trainer.stop()
            if self.verbose > 0:
                print(f'[EarlyStopping] - stopping on epoch {c.epoch}')


