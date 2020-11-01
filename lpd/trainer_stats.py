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

    def add_value(self, value, count):
        self.sum += value * count
        self.count += count
        self.last = value

    def get_mean(self):
        if self.count == 0:
            return 0
        mean = self.sum/self.count
        return mean


class StatsResult():
    def __init__(self, trainer_name, trainer_stats):
        self.trainer_name = trainer_name
        self.loss = trainer_stats.get_loss()
        self.metrics = trainer_stats.get_metrics()

    def __str__(self):
        metrics_str = 'Metrics: no metrics found'
        if self.metrics:
            metrics_str = f'Metrics: {self.metrics}'

        return '------------------\n' \
               'Evaluation Result:\n' \
               f'Trainer name: "{self.trainer_name}"\n' \
               f'Loss: {self.loss}\n' \
               f'{metrics_str}\n' \
               '------------------'


class TrainerStats():
    def __init__(self, metric_name_to_func):
        self.metric_name_to_func = metric_name_to_func
        self.loss_stats = Stats()
        self.metric_name_to_stats = {metric_name:Stats() for metric_name,_ in self.metric_name_to_func.items()}

    def reset(self):
        self.loss_stats.reset()
        for metric_name, stats in self.metric_name_to_stats.items():
            stats.reset()

    def add_loss(self, loss, batch_size):
        self.loss_stats.add_value(loss.item(), batch_size)

    def add_metrics(self, y_pred, y_true, batch_size):
        for metric_name, stats in self.metric_name_to_stats.items():
            metric_func = self.metric_name_to_func[metric_name]
            metric_value = metric_func(y_pred, y_true)
            stats.add_value(metric_value.item(), batch_size)

    def get_loss(self):
        return self.loss_stats.get_mean()

    def get_metrics(self):
        return {metric_name:stats.get_mean() for metric_name, stats in self.metric_name_to_stats.items()}