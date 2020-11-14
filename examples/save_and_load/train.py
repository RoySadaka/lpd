import os
import torch.optim as optim
import torch.nn as nn

from lpd.trainer import Trainer
from lpd.callbacks import SchedulerStep, StatsPrint, ModelCheckPoint, LossOptimizerHandler, CallbackMonitor
from lpd.extensions.custom_schedulers import DoNothingToLR
from lpd.enums import Phase, State, MonitorType, StatsType, MonitorMode, MetricMethod
from lpd.metrics import BinaryAccuracyWithLogits, MetricBase, TruePositives, TrueNegatives, MetricConfusionMatrixBase
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
    num_epochs = 5
    data_loader = eu.examples_data_generator(N, D_in, D_out, binary_out=True)
    data_loader_steps = 100
    return N, D_in, H, D_out, num_epochs, data_loader, data_loader_steps


# LET'S CREATE A CUSTOM (ALTOUGH NOT SO INFORMATIVE) METRIC
class InaccuracyWithLogits(MetricBase):
    def __init__(self):
        super(InaccuracyWithLogits, self).__init__(MetricMethod.MEAN)
        self.bawl = BinaryAccuracyWithLogits() # we exploit BinaryAccuracyWithLogits for the computation

    def __call__(self, y_pred, y_true): # <=== implement this method!
        acc = self.bawl(y_pred, y_true)
        return 1 - acc  # return the inaccuracy


# LET'S CREATE A CUSTOM CONFUSION-MATRIX BASED METRIC
class Truthfulness(MetricConfusionMatrixBase):
    def __init__(self):
        super(Truthfulness, self).__init__(num_classes=2, labels=None,  predictions_to_classes_convertor = None, threshold=0.0)
        self.tp = TruePositives(num_classes=2, threshold=0.0) # we exploit TruePositives for the computation
        self.tn = TrueNegatives(num_classes=2, threshold=0.0) # we exploit TrueNegatives for the computation

    def __call__(self, y_pred, y_true):
        tp_res = self.tp(y_pred, y_true)
        tn_res = self.tn(y_pred, y_true)
        return tp_res + tn_res

def get_trainer_base(D_in, H, D_out):
    device = tu.get_gpu_device_if_available()

    model = eu.get_basic_model(D_in, H, D_out).to(device)

    loss_func = nn.BCEWithLogitsLoss().to(device)
   
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    scheduler = DoNothingToLR() #CAN ALSO USE scheduler=None, BUT DoNothingToLR IS MORE EXPLICIT
    
    metric_name_to_func = {"Accuracy":BinaryAccuracyWithLogits(), 
                           "InAccuracy":InaccuracyWithLogits(), 
                           "TruePositives":TruePositives(num_classes=2, threshold=0.0), 
                           "TrueNegatives":TrueNegatives(num_classes=2, threshold=0.0), 
                           "Truthfulness":Truthfulness()}

    return device, model, loss_func, optimizer, scheduler, metric_name_to_func

def get_trainer(N, D_in, H, D_out, num_epochs, data_loader, data_loader_steps):
    device, model, loss_func, optimizer, scheduler, metric_name_to_func = get_trainer_base(D_in, H, D_out)

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
                    StatsPrint(train_metrics_monitors=[CallbackMonitor(monitor_type=MonitorType.METRIC,
                                                                       stats_type=StatsType.TRAIN,
                                                                       monitor_mode=MonitorMode.MAX,
                                                                       metric_name='Accuracy'),
                                                       CallbackMonitor(monitor_type=MonitorType.METRIC,
                                                                       stats_type=StatsType.TRAIN,
                                                                       monitor_mode=MonitorMode.MIN,
                                                                       metric_name='InAccuracy')],
                               print_confusion_matrix_normalized=True)
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
                      name='Save-And-Load-Example')
    return trainer

def load_trainer(N, D_in, H, D_out, num_epochs, data_loader, data_loader_steps):
    device, model, loss_func, optimizer, scheduler, metric_name_to_func = get_trainer_base(D_in, H, D_out)

    # NOTICE, load_trainer IS A STATIC METHOD IN Trainer CLASS
    loaded_trainer = Trainer.load_trainer(dir_path=save_to_dir,
                                            file_name=trainer_file_name + '_manual_save',
                                            model=model,
                                            device=device,
                                            loss_func=loss_func,
                                            optimizer=optimizer,
                                            scheduler=scheduler,
                                            train_data_loader=data_loader, 
                                            val_data_loader=data_loader,
                                            train_steps=data_loader_steps,
                                            val_steps=data_loader_steps)
    return loaded_trainer

def run():
    params = get_parameters()
    num_epochs = params[4]
    current_trainer = get_trainer(*params)

    current_trainer.summary()

    # TRAINING WILL SAVE current_trainer IN ModelCheckPoint
    current_trainer.train(num_epochs)

    # YOU CAN ALSO SAVE THIS TRAINER MANUALLY, LIKE THE CODE BELOW
    current_trainer.save_trainer(save_to_dir, trainer_file_name + '_manual_save')

    # NOW LETS CREATE A NEW TRAINER FROM THE SAVED FILE,
    loaded_trainer = load_trainer(*params)

    # CONTINUE TRAINING
    loaded_trainer.train(num_epochs)
