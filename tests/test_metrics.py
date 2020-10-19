import unittest
import torch as T
from lpd.metrics import BinaryAccuracy, BinaryAccuracyWithLogits, CategoricalAccuracy, CategoricalAccuracyWithLogits
from torch.autograd import Variable

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
        y_pred = Variable(T.eye(dim))
        y_true = Variable(T.LongTensor(list(range(dim))))
        self.assertEqual(metric(y_pred, y_true), 1.0)
        
        # y_pred:
        # [1,0,0,0]
        # [1,0,0,0]
        # [1,0,0,0]
        # [1,0,0,0]

        # y_true:
        # [0,1,2,3]
        y_pred = Variable(T.zeros(dim, dim))
        y_pred.data[:, 0] = T.ones(dim)
        y_true = Variable(T.LongTensor(list(range(dim))))
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

