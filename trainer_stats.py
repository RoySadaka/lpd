class TrainerStats():
    def __init__(self, round_values_to=None):
        self.sum = 0
        self.count = 0
        self.last = 0
        self.round_values_to = round_values_to

    def add_value(self, value):
        self.sum += value
        self.count += 1
        self.last = value

    def get_mean(self):
        mean = self.sum/self.count
        if self.round_values_to:
            return round(mean, self.round_values_to)
        return mean