class Stats():
    def __init__(self):
        self.sum = 0
        self.count = 0
        self.last = 0
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0
        self.last = 0

    def add_value(self, value):
        self.sum += value
        self.count += 1
        self.last = value

    def get_mean(self):
        if self.count == 0:
            return 0
        mean = self.sum/self.count
        return mean