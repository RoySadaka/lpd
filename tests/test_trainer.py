import os
import unittest

import torch.optim as optim
import torch.nn as nn
from lpd.trainer import Trainer
from lpd.callbacks import StatsPrint, SchedulerStep, LossOptimizerHandler, ModelCheckPoint
from lpd.extensions.custom_schedulers import KerasDecay
from lpd.enums import Phase, State
from lpd.metrics import BinaryAccuracyWithLogits
import lpd.utils.torch_utils as tu
import lpd.utils.general_utils as gu
import examples.utils as eu



class TestTrainer(unittest.TestCase):

    def test_save_and_load(self):
        gu.seed_all(42)  # BECAUSE ITS THE ANSWER TO LIFE AND THE UNIVERSE
        save_to_dir = os.path.dirname(__file__) + '/trainer_checkpoint/'
        trainer_file_name = 'trainer'

        device = tu.get_gpu_device_if_available()

        model = eu.get_basic_model(10, 10, 10).to(device)

        loss_func = nn.BCEWithLogitsLoss().to(device)
    
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        scheduler = KerasDecay(optimizer, 0.0001, last_step=-1)
        
        metric_name_to_func = {"acc":BinaryAccuracyWithLogits()}

        callbacks = [   
                        LossOptimizerHandler(),
                        ModelCheckPoint(checkpoint_dir=save_to_dir, checkpoint_file_name=trainer_file_name, save_best_only=False, save_full_trainer=True),
                        SchedulerStep(apply_on_phase=Phase.BATCH_END, apply_on_states=State.TRAIN),
                        StatsPrint()
                    ]

        
        data_loader = eu.examples_data_generator(10, 10, 10)
        data_loader_steps = 100
        num_epochs = 5

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
                        name='Trainer-Test')
        
        trainer.train(num_epochs)

        loaded_trainer = Trainer.load_trainer(dir_path=save_to_dir,
                                            file_name=trainer_file_name + f'_epoch_{num_epochs}',
                                            model=model,
                                            device=device,
                                            loss_func=loss_func,
                                            optimizer=optimizer,
                                            scheduler=scheduler,
                                            train_data_loader=data_loader, 
                                            val_data_loader=data_loader,
                                            train_steps=data_loader_steps,
                                            val_steps=data_loader_steps)
        
        self.assertEqual(loaded_trainer.epoch, trainer.epoch)
        self.assertListEqual(tu.get_lrs_from_optimizer(loaded_trainer.optimizer), tu.get_lrs_from_optimizer(trainer.optimizer))
        self.assertEqual(loaded_trainer.callbacks[1].monitor._get_best(), trainer.callbacks[1].monitor._get_best())

