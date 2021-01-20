import os
import torch.optim as optim
import torch.nn as nn

from lpd.trainer import Trainer
from lpd.predictor import Predictor
from lpd.callbacks import SchedulerStep, StatsPrint, ModelCheckPoint, LossOptimizerHandler, CallbackMonitor
from lpd.extensions.custom_schedulers import DoNothingToLR
from lpd.enums import Phase, State, MonitorType, StatsType, MonitorMode
from lpd.metrics import BinaryAccuracyWithLogits, MetricBase
import lpd.utils.torch_utils as tu
import lpd.utils.general_utils as gu
import examples.utils as eu

gu.seed_all(42)  # BECAUSE ITS THE ANSWER TO LIFE AND THE UNIVERSE

save_to_dir = os.path.dirname(__file__) + '/trainer_checkpoint/'
trainer_file_name = 'trainer'

def get_parameters():
    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 64, 100, 100, 1
    num_epochs = 3
    data_loader = eu.examples_data_generator(N, D_in, D_out, binary_out=True)
    data_loader_steps = 100
    return N, D_in, H, D_out, num_epochs, data_loader, data_loader_steps

def get_trainer_base(D_in, H, D_out):
    device = tu.get_gpu_device_if_available()

    model = eu.get_basic_model(D_in, H, D_out).to(device)

    loss_func = nn.BCEWithLogitsLoss().to(device)
   
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    scheduler = DoNothingToLR() #CAN ALSO USE scheduler=None, BUT DoNothingToLR IS MORE EXPLICIT
    
    metrics = BinaryAccuracyWithLogits(name='Accuracy')

    return device, model, loss_func, optimizer, scheduler, metrics

def get_trainer(N, D_in, H, D_out, num_epochs, data_loader, data_loader_steps):
    device, model, loss_func, optimizer, scheduler, metrics = get_trainer_base(D_in, H, D_out)

    callbacks = [   
                    LossOptimizerHandler(),
                    #ADDING ModelCheckPoint WITH save_full_trainer=True TO SAVE FULL TRAINER
                    ModelCheckPoint(checkpoint_dir=save_to_dir, 
                                    checkpoint_file_name=trainer_file_name, 
                                    callback_monitor=CallbackMonitor(monitor_type=MonitorType.LOSS, 
                                                                     stats_type=StatsType.VAL, 
                                                                     monitor_mode=MonitorMode.MIN),
                                    save_best_only=True, 
                                    save_full_trainer=True),
                    SchedulerStep(),
                    # SINCE ACCURACY NEEDS TO GO UP AND INACCURACY NEEDS TO GO DOWN, LETS DEFINE CallbackMonitors for StatsPrint PER EACH METRIC
                    StatsPrint(train_metrics_monitors=CallbackMonitor(monitor_type=MonitorType.METRIC,
                                                                       stats_type=StatsType.TRAIN,
                                                                       monitor_mode=MonitorMode.MAX,
                                                                       metric_name='Accuracy'))
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
                      name='Train-Save-Load-Predict-Example')
    return trainer

def run():
    N, D_in, H, D_out, num_epochs, data_loader, data_loader_steps = get_parameters()

    current_trainer = get_trainer(N, D_in, H, D_out, num_epochs, data_loader, data_loader_steps)

    # TRAINING WILL SAVE current_trainer IN ModelCheckPoint
    current_trainer.train(num_epochs)

    # YOU CAN ALSO SAVE THIS TRAINER MANUALLY, LIKE THE CODE BELOW
    current_trainer.save_trainer(save_to_dir, trainer_file_name + '_manual_save')

    # NOW LETS CREATE A PREDICTOR FROM THE SAVED FILE
    device = tu.get_gpu_device_if_available()
    model = eu.get_basic_model(D_in, H, D_out).to(device)
    predictor = Predictor(model=model, device=device)

    data_generator_for_predictions = eu.examples_prediction_data_generator(data_loader, data_loader_steps)

    #PREDICT ON A SINGLE SAMPLE
    sample = next(data_generator_for_predictions)[0]
    sample_prediction = predictor.predict_sample(sample)

    # PREDICT ON A SINGLE BATCH
    batch = next(data_generator_for_predictions)
    batch_prediction = predictor.predict_batch(batch)

    # PREDICTION ON A DATA LOADER
    data_loader_predictions = predictor.predict_data_loader(data_generator_for_predictions, data_loader_steps)