from lpd.enums import Phase, State
from lpd.callbacks.callback_base import CallbackBase
from typing import Union, List, Optional


class LossOptimizerHandlerBase(CallbackBase):
    """
        In case LossOptimizerHandler does not suitable for your needs, create your custom
        callback and derive from this class, and implement __call__ .
        There you have full control for handling loss and optimizer

        Args:
            apply_on_phase - see in CallbackBase
            apply_on_states - see in CallbackBase
            round_values_on_print_to - see in CallbackBase
    """


    def __init__(self, apply_on_phase: Phase, 
                    apply_on_states: Union[State, List[State]], 
                    round_values_on_print_to: Optional[int]=None):
        super(LossOptimizerHandlerBase, self).__init__(apply_on_phase, apply_on_states, round_values_on_print_to)