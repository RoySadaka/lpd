from lpd.callbacks.loss_optimizer_handler_base import LossOptimizerHandlerBase
from lpd.enums import Phase, State
from lpd.callbacks.callback_context import CallbackContext
from typing import Union, List, Callable


class LossOptimizerHandler(LossOptimizerHandlerBase):
    """
        The basic loss and optimizer handler to invoke loss.backward(), optimizer.step() and optimizer.zero_grad()

        Args:
            apply_on_phase - see in CallbackBase
            apply_on_states - see in CallbackBase
            loss_handler - a Callable to handle loss.backward(...)
                           if None, will call loss.backward()
            optimizer_step_handler - a Callable to handle optimizer.step(...)
                                     if None, will call optimizer.step()
            optimizer_zero_grad_handler - a Callable to handle optimizer.zero_grad(...)
                                          if None, will call optimizer.zero_grad()
    """

    def __init__(self, 
                    apply_on_phase: Phase=Phase.BATCH_END, 
                    apply_on_states: Union[State, List[State]]=State.TRAIN,
                    loss_handler: Callable[[CallbackContext], None] = None,
                    optimizer_step_handler: Callable[[CallbackContext], None] = None,
                    optimizer_zero_grad_handler: Callable[[CallbackContext], None] = None):
        super(LossOptimizerHandler, self).__init__(apply_on_phase, apply_on_states)
        self.loss_handler = loss_handler or self._default_loss_handler
        self.optimizer_step_handler = optimizer_step_handler or self._default_optimizer_step_handler
        self.optimizer_zero_grad_handler = optimizer_zero_grad_handler or self._default_optimizer_zero_grad_handler

    def _default_loss_handler(self, callback_context: CallbackContext):
        loss = callback_context.train_last_loss
        loss.backward()

    def _default_optimizer_step_handler(self, callback_context: CallbackContext):
        optimizer = callback_context.optimizer
        optimizer.step()

    def _default_optimizer_zero_grad_handler(self, callback_context: CallbackContext):
        optimizer = callback_context.optimizer
        optimizer.zero_grad()

    def __call__(self, callback_context: CallbackContext):
        c = callback_context #READABILITY DOWN THE ROAD

        self.loss_handler(c)

        self.optimizer_step_handler(c)

        self.optimizer_zero_grad_handler(c)