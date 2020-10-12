from lpd.enums import Phase, State, MonitorType, MonitorMode, StatsType
from lpd.callbacks.callback_context import CallbackContext
from typing import Union, List, Optional, Dict

class CallbackBase():
    """
        Agrs:
            apply_on_phase - (lpd.enums.Phase) the phase to invoke this callback
            apply_on_states - (lpd.enums.State) state or list of states to invoke this parameter (under the relevant phase), None will invoke it on all states
            round_values_on_print_to - optional, it will round the numerical values in the prints
    """

    def __init__(self, apply_on_phase: Phase, 
                       apply_on_states: Union[State, List[State]], 
                       round_values_on_print_to: Optional[int]=None):
        self.apply_on_phase = apply_on_phase
        if self.apply_on_phase is None:
            print('[CallbackBase][Error!] - No callback phase was provided')

        self.apply_on_states = apply_on_states
        self.round_values_on_print_to = round_values_on_print_to

    def round_to(self, value: Union[float,list]):
        if self.round_values_on_print_to:
            if isinstance(value, list):
                return [round(v, self.round_values_on_print_to) for v in value]
            if isinstance(value, float):
                return round(value, self.round_values_on_print_to)
        return value

    def should_apply_on_phase(self, callback_context: CallbackContext):
        if isinstance(self.apply_on_phase, Phase):
            return callback_context.trainer_phase == self.apply_on_phase
        raise ValueError('[CallbackBase] - got bad value for apply_on_phase')

    def should_apply_on_state(self, callback_context: CallbackContext):
        if self.apply_on_states is None:
            return True

        if isinstance(self.apply_on_states, list):
            for state in self.apply_on_states:
                if isinstance(state, State):
                    if callback_context.trainer_state == state:
                        return True
            return False

        if isinstance(self.apply_on_states, State):
            return callback_context.trainer_state == self.apply_on_states

        raise ValueError('[CallbackBase] - got bad value for apply_on_states')
