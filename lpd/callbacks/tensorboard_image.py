from typing import Union, List, Optional, Dict, Callable

import torch
from lpd.enums import Phase, State
from lpd.callbacks.callback_base import CallbackBase
from lpd.callbacks.callback_context import CallbackContext
from lpd.input_output_label import InputOutputLabel

class TensorboardImage(CallbackBase):
    """ 
        Writes image entries directly to event files in the summary_writer_dir to be
        consumed by TensorBoard.
        Args:
            apply_on_phase - see in CallbackBase
            apply_on_states - see in CallbackBase
            summary_writer_dir - the folder path to save tensorboard output
            description - the header of the images
            outputs_parser - optional, a function that accepts InputOutputLabel , and returns a tensor to be paseed to "add_image" in SummaryWriter
                             if None, model output will be passed as is
    """

    def __init__(self, 
                 apply_on_phase: Phase=Phase.EPOCH_END, 
                 apply_on_states: Union[State, List[State]]=State.EXTERNAL,
                 summary_writer_dir: str=None,
                 description: str=None,
                 outputs_parser: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]=None,
                ):
        super(TensorboardImage, self).__init__(apply_on_phase, apply_on_states)
        from torch.utils.tensorboard import SummaryWriter # OPTIMIZATION FOR lpd-nodeps
        self.summary_writer_dir = summary_writer_dir
        self.outputs_parser = outputs_parser if outputs_parser else TensorboardImage.default_output_parser
        self.description = description if description else 'Images'

        if self.summary_writer_dir is None:
            raise ValueError("[TensorboardImage] - summary_writer_dir was not provided")
        self.tensorboard_writer = SummaryWriter(summary_writer_dir)
        self.inner_step = 0

    @staticmethod
    def default_output_parser(input_output_label: InputOutputLabel):
        import torchvision # OPTIMIZATION FOR lpd-nodeps
        tb_ready = torchvision.utils.make_grid(input_output_label.output, normalize=True)
        return tb_ready

    def __call__(self, callback_context: CallbackContext):
        c = callback_context #READABILITY DOWN THE ROAD
        state = c.trainer_state

        last_data = c.trainer._last_data[state]
        outputs = self.outputs_parser(last_data)
        self.tensorboard_writer.add_image(f'{state} {self.description}', outputs, global_step=self.inner_step)
        self.inner_step += 1