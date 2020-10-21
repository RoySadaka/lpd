![Logo](https://raw.githubusercontent.com/RoySadaka/ReposMedia/main/lpd/images/logo.png)

# lpd

A Fast, Flexible Trainer with Callbacks and Extensions for PyTorch

``lpd`` derives from the Hebrew word *lapid* (לפיד) which means "torch".



## For latest PyPI stable release 
[![PyPI version](https://badge.fury.io/py/lpd.svg)](https://badge.fury.io/py/lpd) 
[![Downloads](https://pepy.tech/badge/lpd)](https://pepy.tech/project/lpd)
![Liecense](https://img.shields.io/github/license/roysadaka/lpd)
<!-- ![Follow](https://img.shields.io/twitter/follow/roysadaka?label=RoySadaka&style=social) -->


```sh
    pip install lpd
```

## Usage

``lpd`` intended to properly structure your PyTorch model training. The main usages are given below.

### Training your model

```python
    from lpd.trainer import Trainer
    from lpd.enums import Phase, State, MonitorType, MonitorMode, StatsType
    from lpd.callbacks import StatsPrint, ModelCheckPoint, Tensorboard, EarlyStopping, SchedulerStep
    from lpd.extensions.custom_schedulers import KerasDecay
    from lpd.metrics import BinaryAccuracyWithLogits
    from lpd.utils.torch_utils import get_gpu_device_if_available
    from lpd.utils.general_utils import seed_all

    seed_all(seed=42) # because its the answer to life and the universe

    device = get_gpu_device_if_available() # with fallback to CPU if GPU not avilable
    model = MyModel(config, num_embeddings).to(device) # this is your model class, and its being sent to the relevant device
    optimizer = optim.SGD(params=model.parameters())
    scheduler = KerasDecay(optimizer, decay=0.01, last_step=-1) # decay scheduler using keras formula 
    loss_func = nn.BCEWithLogitsLoss().to(device) # this is your loss class, already sent to the relevant device
    metric_name_to_func = {'acc':BinaryAccuracyWithLogits()} # define your metrics in a dictionary

    # you can use some of the defined callbacks, or you can create your own
    callbacks = [
                SchedulerStep(apply_on_phase=Phase.BATCH_END, apply_on_states=State.TRAIN),
                ModelCheckPoint(checkpoint_dir, checkpoint_file_name, MonitorType.LOSS, StatsType.VAL, MonitorMode.MIN, save_best_only=True), 
                Tensorboard(summary_writer_dir=summary_writer_dir),
                EarlyStopping(patience=10, MonitorType.METRIC, StatsType.VAL, MonitorMode.MAX, metric_name='acc'),
                StatsPrint(apply_on_phase=Phase.EPOCH_END, metric_names=metric_name_to_func.keys())
            ]

    trainer = Trainer(model, 
                      device, 
                      loss_func, 
                      optimizer,
                      scheduler,
                      metric_name_to_func, 
                      train_data_loader,  # DataLoader, Iterable or Generator
                      val_data_loader,    # DataLoader, Iterable or Generator
                      train_steps,
                      val_steps,
                      num_epochs,
                      callbacks,
                      name='Readme-Example')
    
    trainer.train()
```

### Evaluating your model
```python
    trainer.evaluate(test_data_loader, test_steps)
```

### Making predictions
On data loader:
```python
    predictions = trainer.predict(data_loader, steps)
```

On batch:
```python
    prediction = trainer.predict_batch(x)
```

## TrainerStats
``Trainer`` tracks stats for `train/validate/test` and you can access them in your custom callbacks
or any other place that has access to your trainer.

Here are some examples
```python
    train_loss = trainer.train_stats.get_loss()         # the mean of the last epoch's train losses
    val_loss = trainer.val_stats.get_loss()             # the mean of the last epoch's validation losses
    test_loss = trainer.test_stats.get_loss()           # the mean of the test losses (available only after calling evaluate)

    train_metrics = trainer.train_stats.get_metrics()   # dict(metric_name, mean(values)) of the current epoch in train state
    val_metrics = trainer.val_stats.get_metrics()       # dict(metric_name, mean(values)) of the current epoch in validation state
    test_metrics = trainer.test_stats.get_metrics()     # dict(metric_name, mean(values)) of the test (available only after calling evaluate)
```


## Callbacks
Some common callbacks are available under ``lpd.callbacks``.

Notice that ``apply_on_phase`` (``lpd.enums.Phase``) will determine the execution phase,

and that ``apply_on_states`` (``lpd.enums.State`` or ``list(lpd.enums.State)``) will determine the execution states

These are the current available phases and states, more might be added in future releases
```python
        State.EXTERNAL
        Phase.TRAIN_BEGIN
        # train loop:
            Phase.EPOCH_BEGIN

            State.TRAIN
            # batches loop:
                Phase.BATCH_BEGIN
                # batch
                Phase.BATCH_END
            State.VAL
            # batches loop:
                Phase.BATCH_BEGIN
                # batch
                Phase.BATCH_END
            State.EXTERNAL

            Phase.EPOCH_END
        Phase.TRAIN_END
```

Evaluation phases and states will behave as follow
```python
        State.EXTERNAL
        Phase.TEST_BEGIN
        State.TEST
        # batches loop:
            Phase.BATCH_BEGIN
            # batch
            Phase.BATCH_END
        State.EXTERNAL
        Phase.TEST_END
```


Predict phases and states will behave as follow
```python
        State.EXTERNAL
        Phase.PREDICT_BEGIN
        State.PREDICT
        # batches loop:
            Phase.BATCH_BEGIN
            # batch
            Phase.BATCH_END
        State.EXTERNAL
        Phase.PREDICT_END
```

With phases and states, you have full control over the timing of your callbacks,

### StatsPrint Callback
Below is an output example for ``StatsPrint`` callback that will print an epoch summary at the end of every epoch

![EpochSummary](https://raw.githubusercontent.com/RoySadaka/ReposMedia/main/lpd/images/epoch_summary.png)


### ModelCheckPoint Callback
Saving a checkpoint when a monitored loss/metric has improved.
The callback will save the model, optimizer, scheduler, and epoch number.
You can also configure it to save Full Trainer.

For example, ModelCheckPoint that will save a new *full trainer checkpoint* every time the validation metric_name ``my_metric``
is getting higher than highest value so far.

```python
    ModelCheckPoint(checkpoint_dir, 
                    checkpoint_file_name, 
                    monitor_type=MonitorType.METRIC, 
                    stats_type=StatsType.VAL, 
                    monitor_mode=MonitorMode.MAX, 
                    save_best_only=False, 
                    metric_name='my_metric',
                    save_full_trainer=True)
```

### EarlyStopping Callback
Stops the trainer when a monitored loss/metric has stopped improving.
For example, EarlyStopping that will monitor at the end of every epoch, and stop the trainer if the validation loss didn't improve (decrease) for the last 10 epochs.
```python
    EarlyStopping(apply_on_phase=Phase.EPOCH_END, 
                  apply_on_states=State.EXTERNAL,
                  patience=10, 
                  monitor_type=MonitorType.LOSS, 
                  stats_type=StatsType.VAL, 
                  monitor_mode=MonitorMode.MIN)
```

### SchedulerStep Callback

Will invoke ``step()`` on your scheduler in the desired phase and state.

For example, SchedulerStep callback to invoke ``scheduler.step()`` at the end of every batch, and only in train state (as opposed to validation and test)
then define your SchedulerStep callback like so:
```python
    from lpd.callbacks import SchedulerStep
    from lpd.enums import Phase, State
    SchedulerStep(apply_on_phase=Phase.BATCH_END, apply_on_states=State.TRAIN)
```
In case you need it on validation state as well, pass a list for ``apply_on_states`` like so:
```python
    SchedulerStep(apply_on_phase=Phase.BATCH_END, apply_on_states=[State.TRAIN, State.VAL])
```


### Tensorboard Callback
Will export the loss and the metrics at a given phase and state, in a format that can be viewed on Tensorboard 
```python
    Tensorboard(apply_on_phase=Phase.EPOCH_END, 
                apply_on_states=State.EXTERNAL, 
                summary_writer_dir=dir_path)
```


### CollectOutputs Callback
In case you want to collect the outputs of any given state during training
```python
    CollectOutputs(apply_on_phase=Phase.BATCH_END, apply_on_states=State.VAL)
```
CollectOutputs is automatically used in ``trainer.predict(...)`` to collect the predictions


### Create your custom callbacks

```python
    from lpd.enums import Phase, State
    from lpd.callbacks import CallbackBase

    class MyAwesomeCallback(CallbackBase):
        def __init__(self, apply_on_phase=Phase.BATCH_END, apply_on_states=[State.TRAIN, State.VAL]):
            # make sure to call init parent class
            super(MyAwesomeCallback, self).__init__(apply_on_phase, apply_on_states)

        def __call__(self, callback_context): # <=== implement this method!
            # your implementation here
            # using callback_context, you can access anything in your trainer
            # below are some examples to get the hang of it
            val_loss = callback_context.val_stats.get_loss()
            train_loss = callback_context.train_stats.get_loss()
            train_metrics = callback_context.train_stats.get_metrics()
            val_metrics = callback_context.val_stats.get_metrics()
            opt = callback_context.trainer.optimizer
            scheduler = callback_context.trainer.scheduler

            if val_loss < 0.0001:
                # you can also mark the trainer to STOP training by calling stop()
                callback_context.trainer.stop()
```

Lets expand ``MyAwesomeCallback`` with ``CallbackMonitor`` to track if our validation loss is getting better
```python
    from lpd.callbacks import CallbackBase, CallbackMonitor # <== CallbackMonitor added
    from lpd.enums import Phase, State, MonitorType, StatsType, MonitorMode # <== added few needed enums to configure CallbackMonitor

    class MyAwesomeCallback(CallbackBase):
        def __init__(self, apply_on_phase=Phase.BATCH_END, apply_on_states=[State.TRAIN, State.VAL]):
            super(MyAwesomeCallback, self).__init__(apply_on_phase, apply_on_states)
            
            # adding CallbackMonitor to track VAL LOSS with regards to MIN (lower is better) and patience or 20 epochs
            self.val_loss_monitor = CallbackMonitor(patience=20, MonitorType.LOSS, StatsType.VAL, MonitorMode.MIN)

        def __call__(self, callback_context: CallbackContext): # <=== implement this method!
            # same as before, using callback_context, you can access anything in your trainer
            train_metrics = callback_context.train_stats.get_metrics()
            val_metrics = callback_context.val_stats.get_metrics()

            # invoke track() method on your monitor and pass callback_context as parameter
            # since you configured your val_loss_monitor, it will get the relevant parameters from callback_context
            monitor_result = self.val_loss_monitor.track(callback_context)

            # monitor_result (lpd.callbacks.CallbackMonitorResult) contains lots of informative properties
            # for example lets check the status of the patience countdown

            if monitor_result.has_patience():
                print(f'[MyAwesomeCallback] - patience left: {monitor_result.patience_left}')

            # Or, let's stop the trainer, by calling the trainer.stop()
            # if our monitored value did not improve

            if not monitor_result.has_improved():
                print(f'[MyAwesomeCallback] - {monitor_result.description} has stopped improving')
                callback_context.trainer.stop()
```

## Metrics
``lpd.metrics`` provides metrics to check the accuracy of your model, let's create a custom metric using ``MetricBase`` and also show the use of ``BinaryAccuracyWithLogits`` in this example
```python
    from lpd.metrics import BinaryAccuracyWithLogits, MetricBase

    # our custom metric
    class InaccuracyWithLogits(MetricBase):
        def __init__(self):
            self.bawl = BinaryAccuracyWithLogits() # we exploit BinaryAccuracyWithLogits for the computation

        def __call__(self, y_pred, y_true): # <=== implement this method!
            # your implementation here
            acc = self.bawl(y_pred, y_true)
            return 1 - acc  # return the inaccuracy

    # now we can define our metrics and pass them to the trainer
    metric_name_to_func = {'accuracy':BinaryAccuracyWithLogits(), 'inaccuracy':InaccuracyWithLogits()}
``` 

## Save and Load full Trainer
Sometimes you just want to save everything so you can continue training where you left off.

To do so, you may use ``ModelCheckPoint`` for saving full trainer by setting parameter
```python
    save_full_trainer=True
``` 
Or, you can invoke it directly from your trainer
```python
    your_trainer.save_trainer(dir_path, file_name)
``` 

Loading a trainer is as simple as:
```python
    loaded_trainer = Trainer.load_trainer(dir_path,             # the folder where the saved trainer file exists 
                                          trainer_file_name,    # the saved trainer file name 
                                          model,                # state_dict will be loaded
                                          device,
                                          loss_func,            # state_dict will be loaded
                                          optimizer,            # state_dict will be loaded
                                          scheduler,            # state_dict will be loaded
                                          train_data_loader,    # provide new/previous data_loader
                                          val_data_loader,      # provide new/previous data_loader
                                          train_steps,
                                          val_steps)
``` 

### Utils
``lpd.utils`` provides few files (torch_utils, file_utils and general_utils)
For example, a good practice is to use 
```python
    import lpd.utils.general_utils as gu
    gu.seed_all(seed=42)  # because its the answer to life and the universe
```
As early as possible in your code, to make sure that results are reproducible

### Extensions
``lpd.extensions`` provides some custom PyTorch layers, and schedulers, these are just some stuff we like using when we create our models, to gain better flexibility.

So you can use them at your own will.
We will add more extensions from time to time.

## TODOS (more added frequently)
* Add Logger
* Add support for multiple schedulers 
* Add support for multiple losses
* Add colab examples

## Something is missing?! please share with us
You can open an issue, but also feel free to email us at torch.lpd@gmail.com
