import torch

from lpd.enums import MetricMethod
from lpd.metrics import MetricBase


class MockMetric(MetricBase):
    def __init__(self, mock_value, name: str):
        super(MockMetric, self).__init__(name=name, metric_method=MetricMethod.LAST)
        self.mock_value = mock_value

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        return torch.FloatTensor([self.mock_value])

    def set_mock_value(self, mock_value):
        self.mock_value = mock_value