import os
import unittest

import torch.optim as optim
import torch.nn as nn
from lpd.trainer import Trainer
from lpd.predictor import Predictor
from lpd.callbacks import LossOptimizerHandler, ModelCheckPoint, CallbackMonitor
from lpd.metrics import BinaryAccuracyWithLogits
from lpd.enums import MonitorType, MonitorMode, StatsType
import lpd.utils.torch_utils as tu
import examples.utils as eu


class TestPredictor(unittest.TestCase):
    def test_save_and_predict(self):

        save_to_dir = os.path.dirname(__file__) + '/trainer_checkpoint/'
        checkpoint_file_name = 'checkpoint'
        trainer_file_name = 'trainer'

        device = tu.get_gpu_device_if_available()

        model = eu.get_basic_model(10, 10, 10).to(device)

        loss_func = nn.BCEWithLogitsLoss().to(device)
    
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        scheduler = None
        
        metric_name_to_func = {"acc":BinaryAccuracyWithLogits()}

        callbacks = [   
                        LossOptimizerHandler(),
                        ModelCheckPoint(checkpoint_dir=save_to_dir, 
                                        checkpoint_file_name=checkpoint_file_name, 
                                        callback_monitor=CallbackMonitor(monitor_type=MonitorType.LOSS, 
                                                                         stats_type=StatsType.VAL, 
                                                                         monitor_mode=MonitorMode.MIN),
                                        save_best_only=True, 
                                        save_full_trainer=False), 
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
                          name='Predictor-Trainer-Test')

        data_generator_for_predictions = eu.examples_prediction_data_generator(data_loader, data_loader_steps)
        sample = next(data_generator_for_predictions)[0]

        # PREDICT BEFORE TRAIN
        sample_prediction_before_train = trainer.predict_sample(sample)

        trainer.train(num_epochs, verbose=0)
        
        # PREDICT AFTER TRAIN
        sample_prediction_from_trainer = trainer.predict_sample(sample)

        # SAVE THE TRAINER
        trainer.save_trainer(save_to_dir, trainer_file_name)

        #-----------------------------------------------#
        # CREATE PREDICTOR FROM CURRENT TRAINER
        #-----------------------------------------------#
        predictor_from_trainer = Predictor.from_trainer(trainer)

        # PREDICT FROM PREDICTOR
        sample_prediction_from_predictor = predictor_from_trainer.predict_sample(sample)

        self.assertFalse((sample_prediction_before_train==sample_prediction_from_trainer).all())
        self.assertTrue((sample_prediction_from_predictor==sample_prediction_from_trainer).all())



        #-----------------------------------------------#
        # LOAD MODEL CHECKPOINT AS NEW PREDICTOR
        #-----------------------------------------------#
        fresh_device = tu.get_gpu_device_if_available()
        fresh_model = eu.get_basic_model(10, 10, 10).to(fresh_device)
        loaded_predictor = Predictor.from_checkpoint(save_to_dir,
                                                     checkpoint_file_name+'_best_only',
                                                     fresh_model, 
                                                     fresh_device)

        # PREDICT AFTER LOAD
        sample_prediction_from_loaded_predictor = loaded_predictor.predict_sample(sample)

        self.assertFalse((sample_prediction_before_train==sample_prediction_from_trainer).all())
        self.assertTrue((sample_prediction_from_loaded_predictor==sample_prediction_from_trainer).all())



        #-----------------------------------------------#
        # LOAD TRAINER CHECKPOINT AS NEW PREDICTOR
        #-----------------------------------------------#
        fresh_device = tu.get_gpu_device_if_available()
        fresh_model = eu.get_basic_model(10, 10, 10).to(fresh_device)
        loaded_predictor = Predictor.from_checkpoint(save_to_dir,
                                                     trainer_file_name,
                                                     fresh_model, 
                                                     fresh_device)

        # PREDICT AFTER LOAD
        sample_prediction_from_loaded_predictor = loaded_predictor.predict_sample(sample)

        self.assertFalse((sample_prediction_before_train==sample_prediction_from_trainer).all())
        self.assertTrue((sample_prediction_from_loaded_predictor==sample_prediction_from_trainer).all())
