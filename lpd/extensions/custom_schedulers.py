class DoNothingToLR():
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self):
        pass

    def get_last_lr(self):
        return [g['lr'] for g in self.optimizer.param_groups]

    def state_dict(self):
        return {}
    
    def load_state_dict(self, checkpoint):
        pass