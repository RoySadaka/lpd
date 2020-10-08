class CallbackContext():
    #REPRESENTS THE INPUT TO THE CALLBACK, NOTICE, SOME VALUES MIGHT BE NONE, DEPENDING ON THE PHASE/STATE OF THE CALLBACK
    def __init__(self, trainer):
        self.epoch = trainer.epoch
        self.iteration = trainer.iteration
        self.iteration_in_epoch = trainer.iteration_in_epoch

        self.train_stats = trainer.train_stats
        self.val_stats = trainer.val_stats
        self.test_stats = trainer.test_stats

        self.trainer_state = trainer.state
        self.trainer_phase = trainer.phase
        self.trainer = trainer
