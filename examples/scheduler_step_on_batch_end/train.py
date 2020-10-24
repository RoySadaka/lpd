import torch.optim as optim
import torch.nn as nn

from lpd.trainer import Trainer
from lpd.callbacks import StatsPrint, SchedulerStep, LossOptimizerHandler
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

    # LETS ADD A StepLR SCHEDULER 
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, gamma=0.999, step_size=1)
    
    metric_name_to_func = None # THIS EXAMPLE DOES NOT USE METRICS, ONLY LOSS

    # LETS ADD SchedulerStep WITH apply_on_phase=Phase.BATCH_END
    # AND apply_on_states=State.TRAIN
    # IT MEANS THAT THE SchedulerStep WILL BE INVOKED AT THE END OF EVERY BATCH, BUT, WILL ONLY BE APPLIED WHEN 
    # IN TRAIN MODE, AND WILL BE IGNORED IN VAL/TEST MODES
    # NOTICE!!! WE USE verbose=1 TO SEE THE PRINTS FOR THIS EXAMPLE, BUT YOU MIGHT PREFER TO USE verbose=0 or verbose=2
    # BECAUSE ON BATCH LEVEL IT WILL PRINT A LOT 
    callbacks = [   
                    LossOptimizerHandler(),
                    SchedulerStep(apply_on_phase=Phase.BATCH_END, apply_on_states=State.TRAIN, verbose=1), #CAN ALSO BE IN FORM OF ARRAY - apply_on_states=[State.TRAIN]
                    StatsPrint(apply_on_phase=Phase.EPOCH_END)
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
                      name='Scheduler-Step-On-Batch-Example')
    return trainer

def run():
    gu.seed_all(42)  # BECAUSE ITS THE ANSWER TO LIFE AND THE UNIVERSE

    N, D_in, H, D_out, num_epochs, data_loader, data_loader_steps = get_parameters()

    trainer = get_trainer(N, D_in, H, D_out, data_loader, data_loader_steps)
    
    trainer.summary()

    #WE NOW EXPECT SchedulerStep TO INVOKE StepLR AT THE END OF EVERY BATCH
    trainer.train(num_epochs)

    trainer.evaluate(data_loader, data_loader_steps)