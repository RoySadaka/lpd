import unittest
import torch as T
from lpd.metrics import BinaryAccuracy, BinaryAccuracyWithLogits, CategoricalAccuracy, CategoricalAccuracyWithLogits, TopKCategoricalAccuracy

class TestMetrics(unittest.TestCase):

    def test_binary_accuracy(self):
        metric = BinaryAccuracy()

        y_pred = T.Tensor([0.,0.,0.,1.])
        y_true = T.Tensor([0.,0.,0.,1.])
        self.assertEqual(metric(y_pred, y_true), 1.0)

        y_pred = T.Tensor([1.,0.,0.,0.])
        y_true = T.Tensor([0.,0.,0.,1.])
        self.assertEqual(metric(y_pred, y_true), 0.5)

        y_pred = T.Tensor([0.,0.,0.,0])
        y_true = T.Tensor([1.,1.,1.,1.])
        self.assertEqual(metric(y_pred, y_true), 0.0)

        y_pred = T.Tensor([[0.,0.],[1.,0.]])
        y_true = T.Tensor([[1.,0.],[1.,0.]])
        self.assertEqual(metric(y_pred, y_true), 0.75)

    def test_binary_accuracy_with_logits(self):
        metric = BinaryAccuracyWithLogits()

        y_pred = T.Tensor([-0.3,-0.3,-0.3,1.7])
        y_true = T.Tensor([0.0,0.0,0.0,1.0])
        self.assertEqual(metric(y_pred, y_true), 1.0)

        y_pred = T.Tensor([1.7,-0.3,-0.3,-0.3])
        y_true = T.Tensor([0.0,0.0,0.0,1.0])
        self.assertEqual(metric(y_pred, y_true), 0.5)

        y_pred = T.Tensor([-0.3,-0.3,-0.3,-0.3])
        y_true = T.Tensor([1.0,1.0,1.0,1.0])
        self.assertEqual(metric(y_pred, y_true), 0.0)

        y_pred = T.Tensor([[-0.3,-0.3],[1.7,-0.3]])
        y_true = T.Tensor([[1.0,0.0],[1.0,0.0]])
        self.assertEqual(metric(y_pred, y_true), 0.75)

    def test_categorical_accuracy(self):
        dim = 4
        metric = CategoricalAccuracy()

        # y_pred:
        # [1,0,0,0]
        # [0,1,0,0]
        # [0,0,1,0]
        # [0,0,0,1]

        # y_true:
        # [0,1,2,3]
        y_pred = T.eye(dim)
        y_true = T.LongTensor(list(range(dim)))
        self.assertEqual(metric(y_pred, y_true), 1.0)
        
        # y_pred:
        # [1,0,0,0]
        # [1,0,0,0]
        # [1,0,0,0]
        # [1,0,0,0]

        # y_true:
        # [0,1,2,3]
        y_pred = T.zeros(dim, dim)
        y_pred.data[:, 0] = T.ones(dim)
        y_true = T.LongTensor(list(range(dim)))
        self.assertEqual(metric(y_pred, y_true), 0.25)

    def test_categorical_accuracy_with_logits(self):
        dim = 4
        metric = CategoricalAccuracyWithLogits()

        y_pred = T.Tensor([[5,1,1],[2,9,1],[2,3,6]])
        y_true = T.Tensor([0,1,2])
        self.assertEqual(metric(y_pred, y_true), 1.0)

        y_pred = T.Tensor([[5,1,1],[2,9,1],[5,1,1]])
        y_true = T.Tensor([0,1,2])
        self.assertEqual(metric(y_pred, y_true), 2./3.)

        y_pred = T.Tensor([[5,1,1],[2,1,9],[5,1,1]])
        y_true = T.Tensor([0,1,2])
        self.assertEqual(metric(y_pred, y_true), 1./3.)

        y_pred = T.Tensor([[2,5,1],[2,1,9],[5,1,1]])
        y_true = T.Tensor([0,1,2])
        self.assertEqual(metric(y_pred, y_true), 0.0)

    def test_top_k_categorical_accuracy(self):
        # K = 1
        metric = TopKCategoricalAccuracy(k=1)

        y_pred = T.Tensor([[0.1, 0.9, 0.8], [0.05, 0.95, 0], [0.05, 0.95, 0]])
        y_true = T.Tensor([1,1,1])
        self.assertEqual(metric(y_pred, y_true), 1.0)

        y_pred = T.Tensor([[0.1, 0.9, 0.5], [0.05, 0.8, 0.]])
        y_true = T.Tensor([2,1])
        self.assertEqual(metric(y_pred, y_true), 0.5)

        y_pred = T.Tensor([[0.1, 0.3, 0.8], [0.05, 0.2, 0.3], [0.8, 0.4, 0.3]])
        y_true = T.Tensor([1,1,0])
        self.assertEqual(metric(y_pred, y_true), 1/3)

        y_pred = T.Tensor([[0.1, 0.3, 0.8], [0.05, 0.2, 0.3], [0.8, 0.4, 0.3], [0.8, 0.4, 0.3]])
        y_true = T.Tensor([1,1,0,2])
        self.assertEqual(metric(y_pred, y_true), 0.25)


        # K = 2
        metric = TopKCategoricalAccuracy(k=2)

        y_pred = T.Tensor([[0.1, 0.9, 0.8], [0.05, 0.95, 0], [0.05, 0.95, 0]])
        y_true = T.Tensor([1,1,1])
        self.assertEqual(metric(y_pred, y_true), 1.0)

        y_pred = T.Tensor([[0.1, 0.9, 0.01], [0.05, 0.8, 0.]])
        y_true = T.Tensor([2,1])
        self.assertEqual(metric(y_pred, y_true), 0.5)

        y_pred = T.Tensor([[0.1, 0.3, 0.8], [0.05, 0.2, 0.3], [0.3, 0.4, 0.8]])
        y_true = T.Tensor([1,1,0])
        self.assertEqual(metric(y_pred, y_true), 2/3)

        y_pred = T.Tensor([[0.3, 0.1, 0.8], [0.5, 0.2, 0.3], [0.8, 0.4, 0.3], [0.8, 0.4, 0.3]])
        y_true = T.Tensor([1,1,0,2])
        self.assertEqual(metric(y_pred, y_true), 0.25)