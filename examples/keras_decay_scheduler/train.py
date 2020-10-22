import torch as T
import torch.nn as nn
import torch.optim as optim

from lpd.trainer import Trainer
from lpd.callbacks import StatsPrint, SchedulerStep, LossOptimizerHandler
from lpd.extensions.custom_schedulers import KerasDecay
from lpd.enums import Phase, State 
import lpd.utils.torch_utils as tu
import lpd.utils.general_utils as gu
import examples.utils as eu
from lpd.metrics import CategoricalAccuracyWithLogits


def get_parameters():
    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 64, 1000, 100, 10
    num_epochs = 5
    data_loader = eu.examples_data_generator(N, D_in, D_out, category_out=True)
    data_loader_steps = 100
    return N, D_in, H, D_out, num_epochs, data_loader, data_loader_steps

def get_trainer(N, D_in, H, D_out, num_epochs, data_loader, data_loader_steps):

    device = tu.get_gpu_device_if_available()

    model = eu.get_basic_model(D_in, H, D_out).to(device)

    loss_func = nn.CrossEntropyLoss().to(device)
   
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    # HERE WE USE KerasDecay, IT WILL DECAY THE LEARNING-RATE USING
    # THE FORMULA USED IN KERAS:
    # LR = INIT_LR * (1./(1. + decay * step))
    # NOTICE THAT step CAN BE BATCH/EPOCH
    # WE WILL RUN IT ON EPOCH LEVEL IN THIS EXAMPLE, SO WE EXPECT WITH DECAY = 0.01:
    
    # EPOCH 0 LR: 0.1 <--- THIS IS THE STARTING POINT
    # EPOCH 1 LR: 0.1 * (1./(1. + 0.01 * 1)) = 0.09900990099
    # EPOCH 2 LR: 0.1 * (1./(1. + 0.01 * 2)) = 0.09803921568
    # EPOCH 3 LR: 0.1 * (1./(1. + 0.01 * 3)) = 0.09708737864
    # EPOCH 4 LR: 0.1 * (1./(1. + 0.01 * 4)) = 0.09615384615
    # EPOCH 5 LR: 0.1 * (1./(1. + 0.01 * 5)) = 0.09523809523
    scheduler = KerasDecay(optimizer, decay=0.01, last_step=-1) 
    
    metric_name_to_func = {'acc':CategoricalAccuracyWithLogits()}

    callbacks = [   
                    LossOptimizerHandler(),
                    SchedulerStep(apply_on_phase=Phase.EPOCH_END,
                                  apply_on_states=State.EXTERNAL,
                                  verbose=1),                        #LET'S PRINT TO SEE THE ACTUAL CHANGES
                    StatsPrint(metric_names=metric_name_to_func.keys())
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
                      name='Keras-Decay-Example')
    return trainer

def run():
    gu.seed_all(42)  # BECAUSE ITS THE ANSWER TO LIFE AND THE UNIVERSE

    N, D_in, H, D_out, num_epochs, data_loader, data_loader_steps = get_parameters()

    trainer = get_trainer(N, D_in, H, D_out, num_epochs, data_loader, data_loader_steps)
    
    trainer.train()
