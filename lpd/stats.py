class Stats():
    def __init__(self, round_values_to=None):
        self.sum = 0
        self.count = 0
        self.last = 0
        self.round_values_to = round_values_to
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
        if self.round_values_to:
            return round(mean, self.round_values_to)
        return mean