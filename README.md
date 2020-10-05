![Logo](https://raw.githubusercontent.com/RoySadaka/ReposMedia/main/lpd/images/logo.png)

# lpd

A Fast, Flexible Trainer and Extensions for Pytorch

``lpd`` derives from the Hebrew word *lapid* (לפיד) which means "torch".

## For latest PyPI stable release
```sh
    pip install lpd
```

## Usage

``lpd`` intended to properly structure your pytorch model training. The main usages are given below.

### Training your model

```python
    from lpd.trainer import Trainer
    import lpd.utils.torch_utils as tu
    import lpd.utils.general_utils as gu
    import lpd.enums as en 
    from lpd.callbacks import EpochEndStats, ModelCheckPoint, Tensorboard, EarlyStopping, SchedulerStep
    from lpd.extensions.custom_metrics import binary_accuracy_with_logits

    gu.seed_all(seed=42)

    device = tu.get_gpu_device_if_available() # with fallback to CPU if GPU not avilable
    model = TestModel(config, num_embeddings).to(device) #this is your model class, and its being sent to the relevant device
    optimizer = optim.SGD(params=model.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True)
    loss_func = nn.BCEWithLogitsLoss().to(device) #this is your loss class, already sent to the relevant device
    metric_name_to_func = {"acc":binary_accuracy_with_logits} # add as much metrics as you like

    # you can use some of the defined callbacks, or you can create your own
    callbacks = [
                SchedulerStep(scheduler_parameters_func=lambda trainer: trainer.val_stats.get_loss()), # notice lambda for scheduler that takes loss in step()
                ModelCheckPoint(checkpoint_dir, checkpoint_file_name, monitor='val_loss', save_best_only=True, round_values_on_print_to=7), 
                Tensorboard(summary_writer_dir=summary_writer_dir),
                EarlyStopping(patience=10, monitor='val_loss'),
                EpochEndStats(cb_phase=en.CallbackPhase.ON_EPOCH_END, round_values_on_print_to=7) # better to put it last on the list (makes better sense in the log prints)
            ]

    trainer = Trainer(model, 
                      device, 
                      loss_func, 
                      optimizer,
                      scheduler,
                      metric_name_to_func, 
                      train_data_loader,  #iterable or generator
                      val_data_loader,    #iterable or generator
                      train_steps,
                      val_steps,
                      num_epochs,
                      callbacks)
    
    trainer.train()
```

### Evaluating your model
```python
    trainer.evaluate(test_data_loader, test_steps)
```

### TrainerStats
``Trainer`` tracks stats for `train/val/test` and you can access them in your custom callbacks
or any other place you see fit.

Here are some examples
```python
    train_loss = trainer.train_stats.get_loss()         #the mean of the last epoch's train losses
    val_loss = trainer.val_stats.get_loss()             #the mean of the last epoch's val losses

    train_metrics = trainer.train_stats.get_metrics()   #dictionary metric_name->mean of the last epoch's train metrics
    val_metrics = trainer.val_stats.get_metrics()       #dictionary metric_name->mean of the last epoch's val metrics
```


### Callbacks
Some common callbacks are available under ``lpd.callbacks``. 

Notice that ``cb_phase`` (``CallbackPhase`` in ``lpd.enums``) will determine the execution phase,

and that ``apply_on_states`` (``State`` in ``lpd.enums``) will determine the execution state

These are the current available phases and states, more might be added in future releases
```python
    class CallbackPhase(Enum): 
        ON_TRAIN_BEGIN   = 0
        ON_TRAIN_END     = 1
        ON_EPOCH_BEGIN   = 2
        ON_EPOCH_END     = 3
        ON_BATCH_BEGIN   = 4
        ON_BATCH_END     = 5

    class State(Enum):
        EXTERNAL     = 0
        TRAIN        = 1
        VAL          = 2 
        TEST         = 3
```
With phases and states you'll have full control over the timing of your callbacks,

so for example, say you need SchedulerStep callback to control your scheduler,

but only at the end of every batch, and only when in train state (as oppose to validation and test)
then define your SchedulerStep callback like so:
```python
    from lpd.callbacks import SchedulerStep
    import lpd.enums as en
    SchedulerStep(cb_phase=en.CallbackPhase.ON_BATCH_END, apply_on_states=en.State.TRAIN)
```
In case you need it on validation state as well, pass a list for ``apply_on_states`` like so:
```python
    SchedulerStep(cb_phase=en.CallbackPhase.ON_BATCH_END, apply_on_states=[en.State.TRAIN, en.State.VAL])
```
Below is an output example for ``EpochEndStats`` callback that will print an epoch summary at the end of every epoch

![EpochSummary](https://raw.githubusercontent.com/RoySadaka/ReposMedia/main/lpd/images/epoch_summary.png)

You can also create your own custom callbacks

```python
    import lpd.enums as en
    from lpd.callbacks import CallbackBase

    class MyAwesomeCallback(CallbackBase):
        def __init__(self, cb_phase=en.CallbackPhase.ON_EPOCH_END, apply_on_states=en.State.TRAIN):
            super(MyAwesomeCallback, self).__init__(cb_phase, apply_on_states)

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
```

### Utils
``lpd.utils`` provides few utils files (torch_utils, file_utils and general_utils)
For example, a good practice is to use 
```python
    import lpd.utils.general_utils as gu
    gu.seed_all(seed=42)  # because its the answer to life and the universe
```
As early as possible in your code, to make sure that results are reproducible

### Extensions
``lpd.extensions`` provides some custom pytorch layers, these are just some layers we like using when we create our models, to gain better flexibility.

So you can use them at your own will, there youll also find custom metrics and schedulers.
We will add more layers, metrics and schedulers from time to time.


## TODOS (more added frequently)
* Add support for multiple schedulers 
* Add support for multiple losses
* EpochEndStats - save and print best accuracies
* Save trainer in checkpoint to enable loading a model and continue training from last checkpoint
* Add colab examples

## Something is missing?! please share with us
You can open an issue, but also feel free to email us at torch.lpd@gmail.com
