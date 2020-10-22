from lpd.enums import Phase, State, MonitorType, MonitorMode, StatsType
from lpd.callbacks.callback_base import CallbackBase
from lpd.callbacks.callback_context import CallbackContext
from lpd.callbacks.callback_monitor import CallbackMonitorResult
from lpd.utils.torch_utils import get_lrs_from_optimizer
from typing import Union, List, Optional, Dict, Callable

class SchedulerStep(CallbackBase):
    """This callback will invoke a "step()" on the scheduler.

        Agrs:
            apply_on_phase - see in CallbackBase
            apply_on_states - see in CallbackBase
            scheduler_parameters_func - Since some schedulers takes parameters in step(param1, param2...)
                And other schedulers step() are parameterless, provide:
                a function (or lambda) that except CallbackContext and returns whatever information needed,
                e.g. for scheduler that takes val_loss as parameter, initialize like this:
                    SchedulerStep(scheduler_parameters_func=lambda callback_context: callback_context.val_stats.get_loss())
                if your scheduler step does not expect parameters, leave scheduler_parameters_func = None
            verbose - prints visibility, 0=no prints, 1=all prints, 2=errors/warnings only
    """
    def __init__(self, apply_on_phase: Phase=Phase.EPOCH_END, 
                       apply_on_states: Union[State, List[State]]=State.EXTERNAL,
                       scheduler_parameters_func=None,
                       verbose: int=0):
        super(SchedulerStep, self).__init__(apply_on_phase=apply_on_phase, apply_on_states=apply_on_states)
        self.scheduler_parameters_func = scheduler_parameters_func
        self.verbose = verbose

    def __call__(self, callback_context: CallbackContext):
        c = callback_context #READABILITY DOWN THE ROAD

        if self.verbose and c.scheduler is None:
            print('[SchedulerStep] - no scheduler defined in trainer')
            return

        current_lrs = get_lrs_from_optimizer(c.optimizer)

        if self.scheduler_parameters_func:
            c.scheduler.step(self.scheduler_parameters_func(c))
        else:
            c.scheduler.step()

        new_lrs = get_lrs_from_optimizer(c.optimizer)
        
        if self.verbose == 1:
            print(f'[SchedulerStep] - current_lrs: {current_lrs} new_lrs: {new_lrs}')
