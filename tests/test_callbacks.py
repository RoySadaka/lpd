import os
import unittest

import torch.optim as optim
import torch.nn as nn

from lpd.metrics.mock_metric import MockMetric
from lpd.trainer import Trainer
from lpd.callbacks import StatsPrint, SchedulerStep, LossOptimizerHandler, ModelCheckPoint, CallbackMonitor, \
    CallbackContext
from lpd.extensions.custom_schedulers import KerasDecay
from lpd.enums import Phase, State, MonitorType, StatsType, MonitorMode
from lpd.metrics import BinaryAccuracyWithLogits, CategoricalAccuracyWithLogits
import lpd.utils.torch_utils as tu
import lpd.utils.general_utils as gu
import examples.utils as eu
from lpd.utils.threshold_checker import AbsoluteThresholdChecker


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

    def test_did_improve_gradually(self):
        gu.seed_all(42)

        device = tu.get_gpu_device_if_available()

        model = eu.get_basic_model(10, 10, 10).to(device)

        loss_func = nn.CrossEntropyLoss().to(device)

        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        scheduler = KerasDecay(optimizer, 0.0001, last_step=-1)

        metrics = MockMetric(0.0, 'mock_metric')

        callbacks = [
            LossOptimizerHandler()
        ]

        data_loader = eu.examples_data_generator(10, 10, 10, category_out=True)
        data_loader_steps = 1

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
                          name='Trainer-Test')

        mock_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        threshold = 0.99
        sp = StatsPrint(train_metrics_monitors=CallbackMonitor(monitor_type=MonitorType.METRIC,
                                                               stats_type=StatsType.TRAIN,
                                                               monitor_mode=MonitorMode.MAX,
                                                               threshold_checker=AbsoluteThresholdChecker(MonitorMode.MAX,
                                                                                                          threshold),
                                                               metric_name='mock_metric'))

        trainer.train(1)  # IMPROVE inf TO 0.0
        res = sp.train_metrics_monitors[0].track(CallbackContext(trainer))
        assert res.did_improve

        for mock_value in mock_values:
            metrics.set_mock_value(mock_value)
            trainer.train(1)
            res = sp.train_metrics_monitors[0].track(CallbackContext(trainer))
            assert not res.did_improve

        metrics.set_mock_value(1.0)  # IMPROVE 0.0 TO 1.0 (> 0.99)
        trainer.train(1)
        res = sp.train_metrics_monitors[0].track(CallbackContext(trainer))
        assert res.did_improve
