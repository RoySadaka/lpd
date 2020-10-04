# THIS EXAMPLE WAS TAKEN FROM:
# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
# AND CONVERTED TO USE lpd

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from lpd.trainer import Trainer
from lpd.callbacks import EpochEndStats, SchedulerStep
from lpd.extensions.custom_schedulers import DoNothingToLR
from lpd.extensions.custom_layers import Dense
import lpd.callbacks as cbs 
import lpd.utils.torch_utils as tu
import lpd.utils.general_utils as gu

# Create random Tensors to hold inputs and outputs
def data_generator(N, D_in, D_out):
    x = T.randn(N, D_in)
    y = T.randn(N, D_out)
    while True:
        yield [x], y #YIELD THE SAME X,y every time

def get_trainer(D_in, H, D_out, data_loader, data_loader_steps, num_epochs):
    device = tu.get_gpu_device_if_available()

    # Use the nn package to define our model and loss function.
    model = nn.Sequential(
                            Dense(D_in, H, use_bias=True, activation=F.relu),
                            Dense(H, D_out, use_bias=True, activation=None)
                        ).to(device)

    loss_func = nn.MSELoss(reduction='sum')
   
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    scheduler = DoNothingToLR(optimizer=optimizer) #CAN ALSO USE scheduler=None, BUT DoNothingToLR IS MORE EXPLICIT
    
    metric_name_to_func = None # THIS EXAMPLE DOES NOT USE METRICS, ONLY LOSS

    callbacks = [   
                    SchedulerStep(),
                    EpochEndStats(cb_phase=cbs.CB_ON_EPOCH_END, round_values_on_print_to=7)
                ]

    trainer = Trainer(model=model, 
                      device=device, 
                      loss_func=loss_func, 
                      optimizer=optimizer,
                      scheduler=scheduler,
                      metric_name_to_func=metric_name_to_func, 
                      train_data_loader=data_loader, 
                      val_data_loader=data_loader,
                      train_steps=data_loader_steps,
                      val_steps=data_loader_steps,
                      num_epochs=num_epochs,
                      callbacks=callbacks)
    return trainer

def run():
    gu.seed_all()

    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 64, 1000, 100, 10
    num_epochs = 50
    data_loader = data_generator(N, D_in, D_out)
    data_loader_steps = 100

    trainer = get_trainer(D_in, H, D_out, data_loader, data_loader_steps, num_epochs)
    
    trainer.summary()

    trainer.train()

    trainer.evaluate(data_loader, data_loader_steps)