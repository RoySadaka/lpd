from torch.optim.lr_scheduler import LambdaLR

class DoNothingToLR():
    def __init__(self, optimizer=None, last_epoch=-1):
        pass
    
    def step(self):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, checkpoint):
        pass


class KerasDecay():
    def __init__(self, optimizer, decay, last_step=-1):
        self.decay = decay
        # THE KERAS DECAY FORMULA
        # LR = INIT_LR * (1.0/(1.0 + decay * step))
        self.fcn = lambda step: (1./(1. + self.decay * step)) 
        self.inner_scheduler = LambdaLR(optimizer, lr_lambda=self.fcn, last_epoch=last_step)

    def step(self):
        self.inner_scheduler.step()

    def state_dict(self):
        return self.inner_scheduler.state_dict()

    def load_state_dict(self, checkpoint):
        self.inner_scheduler.load_state_dict(checkpoint)

