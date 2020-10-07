# THIS EXAMPLE WAS TAKEN FROM:
# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
# AND CONVERTED TO USE lpd

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from lpd.trainer import Trainer
from lpd.callbacks import StatsPrint, SchedulerStep
from lpd.extensions.custom_schedulers import DoNothingToLR
from lpd.extensions.custom_layers import Dense
import lpd.enums as en 
import lpd.utils.torch_utils as tu
import lpd.utils.general_utils as gu

def get_parameters():
    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 64, 1000, 100, 10
    num_epochs = 50
    data_loader = data_generator(N, D_in, D_out)
    data_loader_steps = 100
    return N, D_in, H, D_out, num_epochs, data_loader, data_loader_steps

def data_generator(N, D_in, D_out):
    # Create random Tensors to hold inputs and outputs
    x = T.randn(N, D_in)
    y = T.randn(N, D_out)
    while True:
        yield [x], y #YIELD THE SAME X,y every time

def get_basic_model(D_in, H, D_out):
    return nn.Sequential(
                            Dense(D_in, H, use_bias=True, activation=F.relu),
                            Dense(H, D_out, use_bias=True, activation=None)
                        )

def get_loss(device):
    return nn.MSELoss(reduction='sum').to(device)

def get_trainer(N, D_in, H, D_out, num_epochs, data_loader, data_loader_steps):

    device = tu.get_gpu_device_if_available()

    # Use the nn package to define our model and loss function.
    model = get_basic_model(D_in, H, D_out).to(device)

    loss_func = get_loss(device)
   
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    scheduler = DoNothingToLR(optimizer=optimizer) #CAN ALSO USE scheduler=None, BUT DoNothingToLR IS MORE EXPLICIT
    
    metric_name_to_func = None # THIS EXAMPLE DOES NOT USE METRICS, ONLY LOSS

    callbacks = [   
                    SchedulerStep(),
                    StatsPrint(round_values_on_print_to=7)
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
                      callbacks=callbacks,
                      name='Basic-Example')
    return trainer

def run():
    gu.seed_all(42)  # BECAUSE ITS THE ANSWER TO LIFE AND THE UNIVERSE

    N, D_in, H, D_out, num_epochs, data_loader, data_loader_steps = get_parameters()

    trainer = get_trainer(N, D_in, H, D_out, num_epochs, data_loader, data_loader_steps)
    
    trainer.summary()

    trainer.train()

    trainer.evaluate(data_loader, data_loader_steps)