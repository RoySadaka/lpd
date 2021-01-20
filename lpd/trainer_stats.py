from lpd.enums.metric_method import MetricMethod
from lpd.metrics.metric_base import MetricConfusionMatrixBase
from lpd.metrics.confusion_matrix import ConfusionMatrix



class Stats():
    def __init__(self, metric_method: MetricMethod):
        self.sum = 0
        self.count = 0
        self.last = 0
        self.metric_method = metric_method
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0
        self.last = 0

    def add_value(self, value, count):
        if self.metric_method == MetricMethod.MEAN:
            self.sum += value * count
            self.count += count

        elif self.metric_method == MetricMethod.SUM:
            self.sum += value
            self.count += count
        
        elif self.metric_method == MetricMethod.LAST:
            self.sum = value
            self.count = 1

        self.last = value

    def get_value(self):
        if self.metric_method == MetricMethod.MEAN:
            if self.count == 0:
                return 0
            return self.sum/self.count

        elif self.metric_method == MetricMethod.SUM:
            return self.sum

        elif self.metric_method == MetricMethod.LAST:
            return self.sum

class StatsResult():
    def __init__(self, trainer_name, stats):
        self.trainer_name = trainer_name
        self.loss = stats.get_loss().tolist()
        self.metrics = {name:value.tolist() for name,value in stats.get_metrics().items()}

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
    def __init__(self, metrics):
        self.metrics = metrics
        self.loss_stats = Stats(MetricMethod.MEAN)
        self.metric_name_to_stats = {metric.name:Stats(metric.metric_method) for metric in self.metrics}
        self.confusion_matrix = self._handle_confusion_matrix()

    def _handle_confusion_matrix(self):
        for metric in self.metrics:
            if isinstance(metric, MetricConfusionMatrixBase):
                confusion_matrix = ConfusionMatrix(num_classes=metric.num_classes, 
                                                   labels=metric.labels, 
                                                   predictions_to_classes_convertor=metric.predictions_to_classes_convertor, 
                                                   threshold=metric.threshold)
                # NEED A SINGLE CONFUSION MATRIX PER TRAINER-STATS
                return confusion_matrix
        return None

    def reset(self):
        self.loss_stats.reset()

        for metric_name, stats in self.metric_name_to_stats.items():
            stats.reset()

        if self.confusion_matrix:
            self.confusion_matrix.reset()

    def add_loss(self, loss, batch_size):
        self.loss_stats.add_value(loss.clone().detach(), batch_size)

    def add_metrics(self, y_pred, y_true, batch_size):
        if self.confusion_matrix:
            self.confusion_matrix.update_state(y_pred, y_true)
            MetricConfusionMatrixBase.confusion_matrix_ = self.confusion_matrix

        for metric in self.metrics:
            name = metric.name
            stats = self.metric_name_to_stats[name]
            metric_value = metric(y_pred, y_true)
            stats.add_value(metric_value.clone().detach(), batch_size)

    def get_loss(self):
        return self.loss_stats.get_value()

    def get_metrics(self):
        return {metric_name:stats.get_value() for metric_name, stats in self.metric_name_to_stats.items()}