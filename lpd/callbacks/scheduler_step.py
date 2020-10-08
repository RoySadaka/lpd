from lpd.enums import Phase, State, MonitorType, MonitorMode, StatsType
from lpd.callbacks.callback_base import CallbackBase
from lpd.callbacks.callback_context import CallbackContext
from lpd.callbacks.callback_monitor import CallbackMonitorResult
from typing import Union, List, Optional, Dict

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
    def __init__(self, cb_phase: Phase=Phase.EPOCH_END, 
                       apply_on_states: Union[State, List[State]]=State.EXTERNAL,
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
