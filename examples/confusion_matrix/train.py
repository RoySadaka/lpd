import os
import torch.optim as optim
import torch.nn as nn

from lpd.trainer import Trainer
from lpd.callbacks import SchedulerStep, StatsPrint, ModelCheckPoint, LossOptimizerHandler
from lpd.extensions.custom_schedulers import DoNothingToLR
from lpd.enums import Phase, State, MonitorType, StatsType, MonitorMode
from lpd.metrics import TruePositives, FalsePositives, TrueNegatives, FalseNegatives
import lpd.utils.torch_utils as tu
import lpd.utils.general_utils as gu
import examples.utils as eu

gu.seed_all(42)  # BECAUSE ITS THE ANSWER TO LIFE AND THE UNIVERSE

def get_parameters():
    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out, num_classes = 128, 100, 100, 3,3
    num_epochs = 5
    data_loader = eu.examples_data_generator(N, D_in, D_out, category_out=True)
    data_loader_steps = 100
    return N, D_in, H, D_out, num_epochs, num_classes, data_loader, data_loader_steps


def get_trainer_base(D_in, H, D_out, num_classes):
    device = tu.get_gpu_device_if_available()

    model = eu.get_basic_model(D_in, H, D_out).to(device)

    loss_func = nn.CrossEntropyLoss().to(device)
   
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    scheduler = DoNothingToLR() #CAN ALSO USE scheduler=None, BUT DoNothingToLR IS MORE EXPLICIT
    
    labels = ['Cat', 'Dog', 'Bird']
    metrics = [
                TruePositives(num_classes, labels=labels, threshold = 0),
                FalsePositives(num_classes, labels=labels, threshold = 0),
                TrueNegatives(num_classes, labels=labels, threshold = 0),
                FalseNegatives(num_classes, labels=labels, threshold = 0)
            ]

    return device, model, loss_func, optimizer, scheduler, metrics


def get_trainer(N, D_in, H, D_out, num_epochs, num_classes, data_loader, data_loader_steps):
    device, model, loss_func, optimizer, scheduler, metrics = get_trainer_base(D_in, H, D_out, num_classes)

    callbacks = [   
                    LossOptimizerHandler(),
                    StatsPrint(print_confusion_matrix=True)
                ]

    trainer = Trainer(model=model, 
                      device=device, 
                      loss_func=loss_func, 
                      optimizer=optimizer,
                      scheduler=scheduler,
                      metrics=metrics, 
                      train_data_loader=data_loader, 
                      val_data_loader=data_loader,
                      train_steps=data_loader_steps,
                      val_steps=data_loader_steps,
                      callbacks=callbacks,
                      name='Confusion-Matrix-Example')
    return trainer


def run():
    N, D_in, H, D_out, num_epochs, num_classes, data_loader, data_loader_steps = get_parameters()

    current_trainer = get_trainer(N, D_in, H, D_out, num_epochs, num_classes, data_loader, data_loader_steps)

    current_trainer.train(num_epochs)

