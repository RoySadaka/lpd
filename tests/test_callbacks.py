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


class TestCallbacks(unittest.TestCase):
    def test_stats_print_validations(self):
        # ASSERT INVALID INIT
        self.assertRaises(ValueError, StatsPrint, train_metrics_monitors=CallbackMonitor(monitor_type=MonitorType.METRIC,
                                                                                         stats_type=StatsType.VAL,
                                                                                         monitor_mode=MonitorMode.MAX,
                                                                                         metric_name='Accuracy'))
        # ASSERT VALID INIT
        StatsPrint(train_metrics_monitors=CallbackMonitor(monitor_type=MonitorType.METRIC,
                                                          stats_type=StatsType.TRAIN,
                                                          monitor_mode=MonitorMode.MAX,
                                                          metric_name='Accuracy'))