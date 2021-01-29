import os
import torch
import torch.nn as nn
import torch.optim as optim
from lpd.trainer import Trainer
from lpd.extensions.custom_layers import Dense
from lpd.callbacks import StatsPrint, LossOptimizerHandler, TensorboardImage, Tensorboard, ModelCheckPoint, CallbackMonitor
from lpd.enums import MonitorMode, MonitorType, StatsType
from lpd.extensions.custom_schedulers import DoNothingToLR
import lpd.utils.torch_utils as tu
import lpd.utils.general_utils as gu
import examples.utils as eu

save_to_dir = os.path.dirname(__file__) + '/trainer_checkpoint/'
trainer_file_name = 'trainer'

def data_generator():
    # GENERATE RANDOM IMAGES WITH CxHxW = 1x64x64 in range [-1,1]
    batch_size = 8
    x = (torch.randn(batch_size, 1, 64, 64) * 2.0) - 1.0
    while True:
        yield x, x

class ToyAutoencoder(nn.Module):
    def __init__(self):
        super(ToyAutoencoder, self).__init__()
        self.fc1 = Dense(64, 32, activation=nn.ReLU())
        self.fc2 = Dense(32, 64, activation=nn.Tanh())

    def forward(self, batch_images):
        latent = self.fc1(batch_images)
        out = self.fc2(latent)
        return out


def get_trainer(data_loader):
    base_path = os.path.dirname(__file__) + '/'
    tensorboard_data_dir = base_path + './tensorboard/'

    device = tu.get_gpu_device_if_available()

    model = ToyAutoencoder().to(device)

    loss_func = nn.MSELoss().to(device)
   
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    scheduler = DoNothingToLR() #CAN ALSO USE scheduler=None, BUT DoNothingToLR IS MORE EXPLICIT
    
    metrics = None # THIS EXAMPLE DOES NOT USE METRICS, ONLY LOSS

    callbacks = [   
                    LossOptimizerHandler(),
                    Tensorboard(summary_writer_dir=tensorboard_data_dir),
                    TensorboardImage(summary_writer_dir=tensorboard_data_dir, description='Generated'),
                    ModelCheckPoint(checkpoint_dir=save_to_dir, 
                                    checkpoint_file_name=trainer_file_name, 
                                    callback_monitor=CallbackMonitor(monitor_type=MonitorType.LOSS, 
                                                                     stats_type=StatsType.VAL, 
                                                                     monitor_mode=MonitorMode.MIN),
                                    save_best_only=True, 
                                    save_full_trainer=True),
                    StatsPrint()
                ]

    trainer = Trainer(model=model, 
                      device=device, 
                      loss_func=loss_func, 
                      optimizer=optimizer,
                      scheduler=scheduler,
                      metrics=metrics, 
                      train_data_loader=data_loader, 
                      val_data_loader=data_loader,
                      train_steps=10,
                      val_steps=10,
                      callbacks=callbacks,
                      name='TensorboardImages-Example')
    return trainer

def run():
    data_loader = data_generator()

    gu.seed_all(42)  # BECAUSE ITS THE ANSWER TO LIFE AND THE UNIVERSE

    trainer = get_trainer(data_loader)
    
    trainer.summary()

    trainer.train(5)