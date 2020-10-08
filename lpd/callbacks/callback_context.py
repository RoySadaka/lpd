class CallbackContext():
    #REPRESENTS THE INPUT TO THE CALLBACK, NOTICE, SOME VALUES MIGHT BE NONE, DEPENDING ON THE PHASE OF THE CALLBACK
    def __init__(self, trainer):
        self.epoch = trainer._current_epoch
        self.train_stats = trainer.train_stats
        self.val_stats = trainer.val_stats
        self.test_stats = trainer.test_stats
        self.trainer_state = trainer.state
        self.trainer_phase = trainer.phase
        self.trainer = trainer
