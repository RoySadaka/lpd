import torch.optim as optim
import torch.nn as nn

from lpd.trainer import Trainer
from lpd.callbacks import StatsPrint, SchedulerStep
import lpd.utils.torch_utils as tu
import lpd.utils.general_utils as gu
import examples.utils as eu


def get_parameters():
    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 1, 1000, 100, 10
    num_epochs = 10
    data_loader = eu.examples_data_generator(N, D_in, D_out)
    data_loader_steps = 100
    return N, D_in, H, D_out, num_epochs, data_loader, data_loader_steps

def get_trainer(N, D_in, H, D_out, num_epochs, data_loader, data_loader_steps):

    device = tu.get_gpu_device_if_available()

    model = eu.get_basic_model(D_in, H, D_out).to(device)

    loss_func = nn.MSELoss(reduction='sum').to(device)
   
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    metric_name_to_func = None # THIS EXAMPLE DOES NOT USE METRICS, ONLY LOSS

    callbacks = [   
                    StatsPrint()
                ]

    # HERE WE DEFINE A CLOSURE THAT WILL ACCUMULATE BATCHES UNTIL WE REACH AT LEAST 32 SAMPLES
    # WITH GRADIENTS, AND ONLY THEN, INVOKE STEP AND ZERO GRAD
    # SINCE IN THIS EXAMPLE THE BATCH SIZE WAS SET TO 1, WE EXPECT 31 SKIPS AND INVOCATION ON THE BATCH AFTER
    # NOTICE, WE WILL ALSO INVOKE STEP AND ZERO GRAD ON THE LAST BATCH 
    def opt_step_zero_closure():
        invoke_with_min_samples = 32
        sample_count = 0
        def optimizer_step_and_zero_grad_criteria(trainer):
            nonlocal sample_count
            if trainer.sample_count_in_epoch - sample_count >= invoke_with_min_samples:
                sample_count = trainer.sample_count_in_epoch
                print(f'[Trainer] - Accumulated {invoke_with_min_samples} samples, invoking optimizer step and zero grads')
                return True

            if trainer.iteration_in_epoch == trainer.train_steps:
                sample_count = 0
                print('[Trainer] - Last batch in epoch, invoking optimizer step and zero grads')
                return True

            print(f'[Trainer] - SKIPPING optimizer, {(invoke_with_min_samples - (trainer.sample_count_in_epoch - sample_count))} samples to go')
            return False

        return optimizer_step_and_zero_grad_criteria

    trainer = Trainer(model=model, 
                      device=device, 
                      loss_func=loss_func, 
                      optimizer=optimizer,
                      scheduler=None,
                      metric_name_to_func=metric_name_to_func, 
                      train_data_loader=data_loader,  # DATA LOADER WILL YIELD BATCH SIZE OF 1
                      val_data_loader=data_loader,
                      train_steps=data_loader_steps,
                      val_steps=data_loader_steps,
                      num_epochs=num_epochs,
                      callbacks=callbacks,
                      name='Accumulate-Grads-Example',
                      optimizer_step_and_zero_grad_criteria=opt_step_zero_closure())
    return trainer

def run():
    gu.seed_all(42)  # BECAUSE ITS THE ANSWER TO LIFE AND THE UNIVERSE

    N, D_in, H, D_out, num_epochs, data_loader, data_loader_steps = get_parameters()

    trainer = get_trainer(N, D_in, H, D_out, num_epochs, data_loader, data_loader_steps)

    trainer.train()