from lpd.enums import Phase, State, MonitorType, MonitorMode, StatsType
from lpd.callbacks.callback_context import CallbackContext
from typing import Union, List, Optional, Dict, Iterable

class CallbackBase():
    """
        These are the current available phases and states

            State.EXTERNAL
            Phase.TRAIN_BEGIN
            # train loop:
                Phase.EPOCH_BEGIN

                State.TRAIN
                # batches loop:
                    Phase.BATCH_BEGIN
                    # batch
                    Phase.BATCH_END
                State.VAL
                # batches loop:
                    Phase.BATCH_BEGIN
                    # batch
                    Phase.BATCH_END
                State.EXTERNAL

                Phase.EPOCH_END
            Phase.TRAIN_END

        Evaluation phases and states will behave as follow

            State.EXTERNAL
            Phase.TEST_BEGIN
            State.TEST
            # batches loop:
                Phase.BATCH_BEGIN
                # batch
                Phase.BATCH_END
            State.EXTERNAL
            Phase.TEST_END


        Agrs:
            apply_on_phase - (lpd.enums.Phase) the phase to invoke this callback
            apply_on_states - (lpd.enums.State) state or list of states to invoke this parameter (under the relevant phase), None will invoke it on all states
            round_values_on_print_to - optional, it will round the numerical values in the prints
    """

    def __init__(self, apply_on_phase: Phase, 
                       apply_on_states: Union[State, List[State]], 
                       round_values_on_print_to: Optional[int]=None):
        self.apply_on_phase = apply_on_phase
        self.apply_on_states = self._extract_apply_on_states(apply_on_states)
        self.round_values_on_print_to = round_values_on_print_to
        self._validations()

    def _extract_apply_on_states(self, apply_on_states):
        result = set()
        if isinstance(apply_on_states, State):
            result.add(apply_on_states)
            return result
        elif isinstance(apply_on_states, Iterable):
            for s in apply_on_states:
                if isinstance(s, State) or s is None:
                    result.add(s)
                else:
                    raise ValueError(f'[CallbackBase] - {s} is of type {type(s)}, expected type {State}')
            return result
        elif apply_on_states is None:
            result.add(apply_on_states)
            return result

        raise ValueError(f'[CallbackBase] - got bad value for apply_on_states')

    def _validations(self):
        if self.apply_on_phase is None:
            raise ValueError('[CallbackBase] - No callback phase was provided')
        if None in self.apply_on_states:
            print('[CallbackBase][!] - apply_on_states is None, callback will be applied to all states')

        valid_pairs = {
                        Phase.TRAIN_BEGIN:{None, State.EXTERNAL}, 
                        Phase.TRAIN_END:{None, State.EXTERNAL}, 
                        Phase.EPOCH_BEGIN:{None, State.EXTERNAL}, 
                        Phase.EPOCH_END:{None, State.EXTERNAL}, 
                        Phase.BATCH_BEGIN:{None, State.TRAIN, State.VAL, State.TEST, State.PREDICT}, 
                        Phase.BATCH_END:{None, State.TRAIN, State.VAL, State.TEST, State.PREDICT}, 
                        Phase.TEST_BEGIN:{None, State.EXTERNAL}, 
                        Phase.TEST_END:{None, State.EXTERNAL}, 
                        Phase.PREDICT_BEGIN:{None, State.EXTERNAL}, 
                        Phase.PREDICT_END:{None, State.EXTERNAL}, 
                        }

        if self.apply_on_states is not None:
            for state in self.apply_on_states:
                if state not in valid_pairs[self.apply_on_phase]:
                    valid_print = ' or '.join([str(s) for s in (valid_pairs[self.apply_on_phase]-{None})])
                    raise ValueError(f'[CallbackBase] - State {state} cannot be applied in Phase {self.apply_on_phase}, did you mean {valid_print} ?')

    def get_description(self):
        str_apply_on_states = ','.join([str(s) for s in self.apply_on_states])
        return f'[{self.__class__.__name__}] - apply_on_phase: {self.apply_on_phase}, apply_on_states: {str_apply_on_states}'

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
        if None in self.apply_on_states:
            return True

        for state in self.apply_on_states:
            if callback_context.trainer_state == state:
                return True

        return False