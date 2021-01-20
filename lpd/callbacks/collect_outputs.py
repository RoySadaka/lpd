from lpd.enums import Phase, State
from lpd.callbacks.callback_base import CallbackBase
from lpd.callbacks.callback_context import CallbackContext
from typing import Union, List

class CollectOutputs(CallbackBase):
    """
        This callback will collect outputs per each state, (it is currently used in trainer.predict() method.)
        It will collect the numpy outputs in the defined states to a dictionary (state->outputs)

        Methods:
            get_outputs_for_state - for a given state, returns the collected outputs

        Args:
            apply_on_phase - see in CallbackBase
            apply_on_states - see in CallbackBase
    """

    def __init__(self, 
                    apply_on_phase: Phase, 
                    apply_on_states: Union[State, List[State]]):
        super(CollectOutputs, self).__init__(apply_on_phase, apply_on_states)
        self.state_to_outputs = {}

    def get_outputs_for_state(self, state: State):
        return [data.cpu().numpy() for data in self.state_to_outputs[state]]

    def __call__(self, callback_context: CallbackContext):
        c = callback_context #READABILITY DOWN THE ROAD
        state = c.trainer_state

        if self.should_apply_on_state(c):

            if state not in self.state_to_outputs:
                self.state_to_outputs[state] = []

            last_outputs = c.trainer._last_data[state].output.data
            self.state_to_outputs[state].append(last_outputs)
