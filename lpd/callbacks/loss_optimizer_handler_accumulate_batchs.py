from lpd.callbacks.loss_optimizer_handler_base import LossOptimizerHandlerBase
from lpd.enums import Phase, State
from lpd.callbacks.callback_context import CallbackContext
from typing import Union, List, Callable


class LossOptimizerHandlerAccumulateBatches(LossOptimizerHandlerBase):
    """
        loss and optimizer handler to invoke loss.backward() every batch, 
        but invoke optimizer.step() and optimizer.zero_grad() only after the defined num of batches were accumulated

        Args:
            apply_on_phase - see in CallbackBase
            apply_on_states - see in CallbackBase
            min_num_batchs_before_backprop - the amount of batches to wait before invoking optimizer.step() and optimizer.zero_grad()
            verbose - 0 = no prints, 1 = print when performing step and zero_grad
    """

    def __init__(self, 
                    apply_on_phase: Phase=Phase.BATCH_END, 
                    apply_on_states: Union[State, List[State]]=State.TRAIN,
                    min_num_batchs_before_backprop:int=1,
                    verbose:int=0):
        super(LossOptimizerHandlerAccumulateBatches, self).__init__(apply_on_phase, apply_on_states)
        self.min_num_batchs_before_backprop = min_num_batchs_before_backprop
        self.verbose = verbose
        self._iteration_count = 0

    def _should_backprop(self, callback_context: CallbackContext):
        if callback_context.iteration - self._iteration_count >= self.min_num_batchs_before_backprop:
            self._iteration_count = callback_context.iteration
            return True

        return False

    def __call__(self, callback_context: CallbackContext):
        c = callback_context #READABILITY DOWN THE ROAD

        # loss.backward() computes the derivative of the loss w.r.t. the parameters (or anything requiring gradients) using backpropagation.
        # opt.step() causes the optimizer to take a step based on the gradients of the parameters.
        # opt.zero_grad clears old gradients from the last step (otherwise accumulating the gradients from all loss.backward() calls).

        loss = c.train_last_loss
        loss.backward()

        if self._should_backprop(c):
            if self.verbose:
                print(f'[LossOptimizerHandlerAccumulateBatches] - collected {self.min_num_batchs_before_backprop} batches, calling optimizer step and zero_grad')

            optimizer = c.optimizer
            optimizer.step()

            optimizer.zero_grad()