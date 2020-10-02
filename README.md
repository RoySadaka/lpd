![Logo](https://raw.githubusercontent.com/RoySadaka/lpd/master/images/logo.png)

# lpd

A Fast, Flexible Trainer and Extensions for Pytorch

``lpd`` derives from the Hebrew word *lapid* (לפיד) which means "torch".


## Usage

``lpd`` intended to properly structure your pytorch model training. The main usages are given below.

### Training your model

```python
    from lpd.trainer import Trainer
    import lpd.utils.torch_utils as tu
    from lpd.callbacks import EpochEndStats, ModelCheckPoint, Tensorboard, EarlyStopping
    from lpd.extensions.custom_metrics import binary_accuracy_with_logits

    device = tu.get_gpu_device_if_available()
    model = TestModel(config, num_embeddings).to(device) #this is your model class, already sent to the relevant device
    optimizer = optim.SGD(params=model.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True)
    loss_func = nn.BCEWithLogitsLoss().to(device) #this is your loss class, already sent to the relevant device
    metric_name_to_func = {"acc":binary_accuracy_with_logits} # add as much metrics as you like

    # you can use some of the defined callbacks, or you can create your own
    callbacks = [
                ModelCheckPoint(checkpoint_dir, checkpoint_file_name, monitor='val_loss', save_best_only=True), 
                Tensorboard(summary_writer_dir=summary_writer_dir),
                EarlyStopping(patience=10, monitor='val_loss'),
                EpochEndStats() # better to put it last (makes better sense in the log prints)
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
```

### Evaluating your model
```python
    trainer.evaluate(test_data_loader, test_steps)
```


### Callbacks
Some common callbacks are available under ``lpd.callbacks``. 

Notice that ``cb_phase`` will determine the execution phase.

These are the current available phases, more will be added soon
```python
    CB_ON_TRAIN_BEGIN
    CB_ON_TRAIN_END  
    CB_ON_EPOCH_BEGIN
    CB_ON_EPOCH_END  
```

``EpochEndStats`` callback will print an epoch summary at the end of every epoch

![EpochSummary](https://raw.githubusercontent.com/RoySadaka/lpd/master/images/epoch_summary.png)

You can also create your own callbacks

```python
    import lpd.callbacks as cbs
    from lpd.callbacks import CallbackBase
    class MyAwesomeCallback(CallbackBase):
        def __init__(self, cb_phase=cbs.CB_ON_TRAIN_BEGIN):
            super(MyAwesomeCallback, self).__init__(cb_phase)

        def __call__(self, callback_context):
            # using callback_context, you can access anything in your trainer
            # below are some examples to get the hang of it
            val_loss = callback_context.val_loss_stats.get_mean()
            train_loss = callback_context.train_loss_stats.get_mean()
            train_metrics = callback_context.train_metric_name_to_stats
            val_metrics = callback_context.val_metric_name_to_stats
            opt = callback_context.trainer.optimizer
            scheduler = callback_context.trainer.scheduler
```

### Custom Layers
``lpd.extensions`` provides some custom pytorch layers, this are just some layers we like using when we create our models, to gain better flexibility.

So you can use them at your own will, we will add more layers from time to time.


# TODOS (we add more todos as we go):
* EpochEndStats - save best accuracies
* handle scheduler.step() that takes parameters

# Something is missing?! please share with us
You can open an issue, but also feel free to email us at torch.lpd@gmail.com
