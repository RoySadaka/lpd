from lpd.stats import Stats


class TrainerStats():
    def __init__(self, metric_name_to_func, print_round_values_to=None):
        self.metric_name_to_func = metric_name_to_func
        self.loss_stats = Stats(print_round_values_to)
        self.metric_name_to_stats = {metric_name:Stats(print_round_values_to) for metric_name,_ in self.metric_name_to_func.items()}

    def reset(self):
        self.loss_stats.reset()
        for metric_name, stats in self.metric_name_to_stats.items():
            stats.reset()

    def add_loss(self, loss):
        self.loss_stats.add_value(loss.item())

    def add_metrics(self, y_pred, y_true):
        for metric_name, stats in self.metric_name_to_stats.items():
            metric_func = self.metric_name_to_func[metric_name]
            metric_value = metric_func(y_pred, y_true)
            stats.add_value(metric_value.item())

    def get_loss(self):
        return self.loss_stats.get_mean()

    def get_metrics(self):
        return {metric_name:stats.get_mean() for metric_name, stats in self.metric_name_to_stats.items()}