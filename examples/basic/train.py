import torch as T
import torch.nn as nn
import torch.optim as optim

from lpd.trainer import Trainer
from lpd.callbacks import StatsPrint, SchedulerStep, LossOptimizerHandler
from lpd.extensions.custom_schedulers import DoNothingToLR
from lpd.enums import Phase, State
import lpd.utils.torch_utils as tu
import lpd.utils.general_utils as gu
import examples.utils as eu

def get_parameters():
    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 64, 1000, 100, 10
    num_epochs = 10
    data_loader = eu.examples_data_generator(N, D_in, D_out)
    data_loader_steps = 100
    return N, D_in, H, D_out, num_epochs, data_loader, data_loader_steps


def get_trainer(N, D_in, H, D_out, data_loader, data_loader_steps):

    device = tu.get_gpu_device_if_available()

    model = eu.get_basic_model(D_in, H, D_out).to(device)

    loss_func = nn.MSELoss(reduction='sum').to(device)
   
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    scheduler = DoNothingToLR() #CAN ALSO USE scheduler=None, BUT DoNothingToLR IS MORE EXPLICIT
    
    metric_name_to_func = None # THIS EXAMPLE DOES NOT USE METRICS, ONLY LOSS

    callbacks = [   
                    LossOptimizerHandler(),
                    SchedulerStep(),
                    StatsPrint()
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
                      callbacks=callbacks,
                      name='Basic-Example')
    return trainer

def run():
    gu.seed_all(42)  # BECAUSE ITS THE ANSWER TO LIFE AND THE UNIVERSE

    N, D_in, H, D_out, num_epochs, data_loader, data_loader_steps = get_parameters()

    trainer = get_trainer(N, D_in, H, D_out, data_loader, data_loader_steps)
    
    trainer.summary()

    trainer.train(num_epochs)

    trainer.evaluate(data_loader, data_loader_steps)