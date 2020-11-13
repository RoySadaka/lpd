import os
import unittest

import torch.optim as optim
import torch.nn as nn
from lpd.trainer import Trainer
from lpd.callbacks import StatsPrint, SchedulerStep, LossOptimizerHandler, ModelCheckPoint, CallbackMonitor
from lpd.extensions.custom_schedulers import KerasDecay
from lpd.enums import Phase, State, MonitorType, StatsType, MonitorMode
from lpd.metrics import BinaryAccuracyWithLogits, CategoricalAccuracyWithLogits
import lpd.utils.torch_utils as tu
import lpd.utils.general_utils as gu
import examples.utils as eu



class TestTrainer(unittest.TestCase):

    def test_metric_name_to_func_validation(self):
        device = tu.get_gpu_device_if_available()

        model = eu.get_basic_model(10, 10, 10).to(device)

        loss_func = nn.BCEWithLogitsLoss().to(device)
    
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        scheduler = None
        
        metric_name_to_func = {"acc":lambda x,y: x+y}

        callbacks = [   
                        LossOptimizerHandler(),
                        StatsPrint()
                    ]

        data_loader = eu.examples_data_generator(10, 10, 10)
        data_loader_steps = 100

        # ASSERT BAD VALUE FOR metric_name_to_func
        self.assertRaises(ValueError, Trainer, model=model, 
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

        # ASSERT GOOD VALUE FOR metric_name_to_func
        metric_name_to_func = {"acc":BinaryAccuracyWithLogits()}
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

    def test_save_and_load(self):
        gu.seed_all(42)
        save_to_dir = os.path.dirname(__file__) + '/trainer_checkpoint/'
        trainer_file_name = 'trainer'

        device = tu.get_gpu_device_if_available()

        model = eu.get_basic_model(10, 10, 10).to(device)

        loss_func = nn.CrossEntropyLoss().to(device)
    
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        scheduler = KerasDecay(optimizer, 0.0001, last_step=-1)
        
        metric_name_to_func = {"acc":CategoricalAccuracyWithLogits()}

        callbacks = [   
                        LossOptimizerHandler(),
                        ModelCheckPoint(checkpoint_dir=save_to_dir, 
                                        checkpoint_file_name=trainer_file_name, 
                                        callback_monitor=CallbackMonitor(monitor_type=MonitorType.LOSS, 
                                                                         stats_type=StatsType.VAL, 
                                                                         monitor_mode=MonitorMode.MIN),
                                        save_best_only=False, 
                                        save_full_trainer=True,
                                        verbose=0),
                        SchedulerStep(apply_on_phase=Phase.BATCH_END, apply_on_states=State.TRAIN),
                        StatsPrint()
                    ]

        
        data_loader = eu.examples_data_generator(10, 10, 10, category_out=True)
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
        
        trainer.train(num_epochs, verbose=0)

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

    def test_loss_handler_validation(self):
        device = tu.get_gpu_device_if_available()

        model = eu.get_basic_model(10, 10, 10).to(device)

        loss_func = nn.BCEWithLogitsLoss().to(device)
    
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        scheduler = KerasDecay(optimizer, 0.0001, last_step=-1)
        
        metric_name_to_func = {"acc":BinaryAccuracyWithLogits()}

        callbacks = [   
                        StatsPrint()
                    ]

        
        data_loader = eu.examples_data_generator(10, 10, 10)
        data_loader_steps = 100
        num_epochs = 5
        verbose = 0
        
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
        
        self.assertRaises(ValueError, trainer.train, num_epochs, verbose)

        trainer.callbacks.append(LossOptimizerHandler())
        trainer.train(num_epochs, verbose=0)
