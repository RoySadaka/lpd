import os
import unittest
import torch
import torch.optim as optim
import torch.nn as nn
from lpd.trainer import Trainer
from lpd.predictor import Predictor
from lpd.callbacks import LossOptimizerHandler, ModelCheckPoint, CallbackMonitor
from lpd.metrics import BinaryAccuracyWithLogits
from lpd.enums import MonitorType, MonitorMode, StatsType
import lpd.utils.torch_utils as tu
import examples.utils as eu
from lpd.extensions.custom_layers import Dense


class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()

        self.seq1 = nn.Sequential(
                                        Dense(10, 10, use_bias=True, activation=nn.ReLU()),
                                        Dense(10, 2, use_bias=True, activation=None)
                                    )
        self.seq2 = nn.Sequential(
                                        Dense(10, 10, use_bias=True, activation=nn.ReLU()),
                                        Dense(10, 2, use_bias=True, activation=None)
                                    )
        self.out_fc = Dense(4,1,activation=None)

    def forward(self, x1, x2):
        _x1 = self.seq1(x1)
        _x2 = self.seq2(x2)
        return self.out_fc(torch.cat((_x1,_x2), dim=1))


def data_generator():
    x1 = torch.randn(10,10)
    x2 = torch.randn(10,10)
    y = torch.randn(10,1)
    while True:
        yield [x1,x2], y

class TestPredictor(unittest.TestCase):
    def test_save_and_predict(self):

        save_to_dir = os.path.dirname(__file__) + '/trainer_checkpoint/'
        checkpoint_file_name = 'checkpoint'
        trainer_file_name = 'trainer'

        device = tu.get_gpu_device_if_available()

        model = TestModel().to(device)

        loss_func = nn.BCEWithLogitsLoss().to(device)
    
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        scheduler = None
        
        metrics = BinaryAccuracyWithLogits(name='acc')

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

        data_loader = data_generator()
        data_loader_steps = 100
        num_epochs = 5

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
                          name='Predictor-Trainer-Test')

        x1_x2, y = next(data_loader)
        _ = trainer.predict_batch(x1_x2) # JUST TO CHECK THAT IT FUNCTIONS

        sample = [x1_x2[0][0], x1_x2[1][0]]

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
        fresh_model = TestModel().to(fresh_device)
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
        fresh_model = TestModel().to(fresh_device)
        loaded_predictor = Predictor.from_checkpoint(save_to_dir,
                                                     trainer_file_name,
                                                     fresh_model, 
                                                     fresh_device)

        # PREDICT AFTER LOAD
        sample_prediction_from_loaded_predictor = loaded_predictor.predict_sample(sample)

        self.assertFalse((sample_prediction_before_train==sample_prediction_from_trainer).all())
        self.assertTrue((sample_prediction_from_loaded_predictor==sample_prediction_from_trainer).all())
