import examples.multiple_inputs.train as multiple_inputs_example
import examples.basic.train as basic_example
import examples.scheduler_step_on_batch_end.train as scheduler_step_on_batch_end_example
import examples.data_loader.train as data_loader_example
import examples.save_and_load.train as save_and_load_example
import examples.keras_decay_scheduler.train as keras_decay_scheduler_example
import examples.accumulate_grads.train as accumulate_grads_example
import examples.train_evaluate_predict.train as train_evaluate_predict_example


basic_example.run()
scheduler_step_on_batch_end_example.run()
multiple_inputs_example.run()
data_loader_example.run()
save_and_load_example.run()
keras_decay_scheduler_example.run()
accumulate_grads_example.run()
train_evaluate_predict_example.run()