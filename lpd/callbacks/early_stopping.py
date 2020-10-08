from lpd.enums import Phase, State, MonitorType, MonitorMode, StatsType
from lpd.callbacks.callback_base import CallbackBase
from lpd.callbacks.callback_context import CallbackContext
from lpd.callbacks.callback_monitor import CallbackMonitor, CallbackMonitorResult
from typing import Union, List, Optional, Dict

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
                    cb_phase: Phase=Phase.EPOCH_END, 
                    apply_on_states: Union[State, List[State]]=State.EXTERNAL,
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


