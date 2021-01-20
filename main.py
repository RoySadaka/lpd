from tests.test_metrics import TestMetrics
from tests.test_trainer import TestTrainer
from tests.test_predictor import TestPredictor
from tests.test_callbacks import TestCallbacks
import unittest

import examples.multiple_inputs.train as multiple_inputs_example
import examples.basic.train as basic_example
import examples.scheduler_step_on_batch_end.train as scheduler_step_on_batch_end_example
import examples.data_loader.train as data_loader_example
import examples.save_and_load.train as save_and_load_example
import examples.keras_decay_scheduler.train as keras_decay_scheduler_example
import examples.accumulate_grads.train as accumulate_grads_example
import examples.train_evaluate_predict.train as train_evaluate_predict_example
import examples.predictor.train_save_load_predict as train_save_load_predict_example
import examples.confusion_matrix.train as confusion_matrix_example
import examples.tensorbaord_images.train as tensorbaord_images_example

# EXAMPLES
basic_example.run()
scheduler_step_on_batch_end_example.run()
multiple_inputs_example.run()
data_loader_example.run()
save_and_load_example.run()
keras_decay_scheduler_example.run()
accumulate_grads_example.run()
train_evaluate_predict_example.run()
train_save_load_predict_example.run()
confusion_matrix_example.run()
tensorbaord_images_example.run()
print('----------------------------------------------------------------------')
print('Ran all examples')

# TESTS
unittest.main()