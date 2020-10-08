import torch.optim as optim

from lpd.trainer import Trainer
from lpd.callbacks import StatsPrint, SchedulerStep
from lpd.enums import Phase, State 
import lpd.utils.torch_utils as tu
import lpd.utils.general_utils as gu

# WE WILL USE THE "BASIC TRAIN" EXAMPLE, AND JUST CHANGE THE SCHEDULER AND CALLBACKS 
from examples.basic.train import get_basic_model, get_loss, get_parameters


def get_trainer(N, D_in, H, D_out, num_epochs, data_loader, data_loader_steps):

    device = tu.get_gpu_device_if_available()

    model = get_basic_model(D_in, H, D_out).to(device)

    loss_func = get_loss(device)
   
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # LETS ADD A StepLR SCHEDULER 
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, gamma=0.999, step_size=1)
    
    metric_name_to_func = None # THIS EXAMPLE DOES NOT USE METRICS, ONLY LOSS

    # LETS ADD SchedulerStep WITH cb_phase=Phase.BATCH_END
    # AND apply_on_states=State.TRAIN
    # IT MEANS THAT THE SchedulerStep WILL BE INVOKED AT THE END OF EVERY BATCH, BUT, WILL ONLY BE APPLIED WHEN 
    # IN TRAIN MODE, AND WILL BE IGNORED IN VAL/TEST MODES
    callbacks = [   
                    SchedulerStep(cb_phase=Phase.BATCH_END, apply_on_states=State.TRAIN), #CAN ALSO BE apply_on_states=[State.TRAIN]
                    StatsPrint(cb_phase=Phase.EPOCH_END)
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
                      name='Scheduler-Step-On-Batch-Example')
    return trainer

def run():
    gu.seed_all(42)  # BECAUSE ITS THE ANSWER TO LIFE AND THE UNIVERSE

    N, D_in, H, D_out, num_epochs, data_loader, data_loader_steps = get_parameters()

    trainer = get_trainer(N, D_in, H, D_out, num_epochs, data_loader, data_loader_steps)
    
    trainer.summary()

    #WE NOW EXPECT SchedulerStep TO INVOKE StepLR AT THE END OF EVERY BATCH
    trainer.train()

    trainer.evaluate(data_loader, data_loader_steps)