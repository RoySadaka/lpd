import unittest
import torch as T
import torch.nn as nn
from lpd.metrics import BinaryAccuracy, BinaryAccuracyWithLogits, CategoricalAccuracy, CategoricalAccuracyWithLogits
from lpd.metrics import TopKCategoricalAccuracy, TruePositives, TrueNegatives, FalseNegatives, FalsePositives, MetricConfusionMatrixBase
from lpd.metrics.confusion_matrix import ConfusionMatrix
from lpd.enums import ConfusionMatrixBasedMetric


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

    def test_confusion_matrix(self):
        # WE TEST ALL KINDS OF CLASSIFICATION OUTPUTS, THERE MIGHT BE MORE, 
        # THATS WHY THERE IS THE predictions_to_classes_convertor Parameter
        # WHERE YOU CAN PASS YOUR OWN FUNCTION TO CONVERT y_pred TO CLASS INDICES

        #-------------------- BINARY SINGE DIGIT MODE - LEN(SHAPE) = 1 --------------------#
        # DEFINE FUNCTIONS TO CONVERT y_pred TO CLASSES

        labels = ["a", "b"]
        num_classes = len(labels)
        actual    = T.Tensor([1,1,1,1,0,0,0,0])
        predicted = T.Tensor([1,1,1,1,0,0,0,1])

        metric = ConfusionMatrix(num_classes, labels=labels)
        metric.update_state(predicted, actual)

        conf = metric.get_confusion_matrix()
        self.assertTrue((conf==T.Tensor([[3,0],[1,4]])).all())



        #-------------------- BINARY SINGE DIGIT MODE WITH LOGITS - LEN(SHAPE) = 1 --------------------#

        labels = ["a", "b"]
        num_classes = len(labels)
        actual    = T.Tensor([1,1,1,1,0,0,0,0])
        predicted = T.Tensor([1,1,1,1,-1,-1,-1,1])

        metric = ConfusionMatrix(num_classes, labels=labels, threshold=0)
        metric.update_state(predicted, actual)

        conf = metric.get_confusion_matrix()
        self.assertTrue((conf==T.Tensor([[3,0],[1,4]])).all())


        #-------------------- BINARY SINGE DIGIT MODE - LEN(SHAPE) = 2 --------------------#

        labels = ["a", "b"]
        num_classes = len(labels)
        actual    = T.Tensor([0,0,0,0,1,1,1,1])
        predicted = T.Tensor([[0],[0],[0],[0],[1],[1],[1],[0]])

        metric = ConfusionMatrix(num_classes, labels=labels)
        metric.update_state(predicted, actual)

        conf = metric.get_confusion_matrix()
        self.assertTrue((conf==T.Tensor([[4,1],[0,3]])).all())


        #-------------------- CLASSIFICATION 2 CLASSES --------------------#

        labels = ["a", "b"]
        num_classes = len(labels)
        actual    = T.Tensor([0,0,0,0,1,1,1,1])
        predicted = T.Tensor([[1,0],[1,0],[1,0],[1,0],[0,1],[0,1],[0,1],[1,0]])

        metric = ConfusionMatrix(num_classes, labels=labels)
        metric.update_state(predicted, actual)

        conf = metric.get_confusion_matrix()
        self.assertTrue((conf==T.Tensor([[4,1],[0,3]])).all())


        #-------------------- CLASSIFICATION 3 CLASSES --------------------#

        labels = ["a", "b", "z"]
        num_classes = len(labels)
        emb = nn.Embedding(num_classes,num_classes)
        emb.weight.data = T.eye(num_classes)
        actual    = T.Tensor([0,0,0,0,1,1,1,1])
        predicted = T.Tensor([0,0,0,0,1,1,1,2]).long()
        predicted = emb(predicted)

        metric = ConfusionMatrix(num_classes, labels=labels)
        metric.update_state(predicted, actual)

        conf = metric.get_confusion_matrix()
        self.assertTrue((conf==T.Tensor([[4,0,0],[0,3,0],[0,1,0]])).all())


        #-------------------- CLASSIFICATION 3 CLASSES --------------------#

        labels = ["a", "b", "c"]
        num_classes = len(labels)
        emb = nn.Embedding(num_classes,num_classes)
        emb.weight.data = T.eye(num_classes)
        actual      = T.Tensor([0, 1, 2, 2, 1, 2, 2, 1, 0, 0, 1, 0, 1, 2, 0, 1, 2])
        predicted   = T.Tensor([0, 1, 1, 2, 0, 2, 0, 1, 2, 0, 1, 1, 1, 2, 0, 0, 2]).long()
        predicted  = emb(predicted)

        metric = ConfusionMatrix(num_classes, labels=labels)
        metric.update_state(predicted, actual)

        conf = metric.get_confusion_matrix()
        self.assertTrue((conf==T.Tensor([[3,2,1],[1,4,1],[1,0,4]])).all())


        #-------------------- CLASSIFICATION 4 CLASSES --------------------#
        labels = ["a", "b", "z", "x"]
        num_classes = len(labels)
        emb = nn.Embedding(num_classes,num_classes)
        emb.weight.data = T.eye(num_classes)
        actual    = T.Tensor([0,0,0,2,2,1,1,1])
        predicted = T.Tensor([0,0,0,0,1,1,1,3]).long()
        predicted  = emb(predicted)

        metric = ConfusionMatrix(num_classes, labels=labels)
        metric.update_state(predicted, actual)

        conf = metric.get_confusion_matrix()
        self.assertTrue((conf==T.Tensor([[3,0,1,0],[0,2,1,0],[0,0,0,0],[0,1,0,0]])).all())


        #-------------------- LOGITS - CLASSIFICATION 4 CLASSES --------------------#

        labels = ["a", "b", "z", "x"]
        num_classes = len(labels)
        emb = nn.Embedding(num_classes,num_classes)
        emb.weight.data = (T.eye(num_classes) * 0.6) + 0.2
        actual    = T.Tensor([0,0,0,2,2,1,1,1])
        predicted = T.Tensor([0,0,0,0,1,1,1,3]).long()
        predicted  = emb(predicted)

        metric = ConfusionMatrix(num_classes, labels=labels, threshold=0.75)
        metric.update_state(predicted, actual)

        conf = metric.get_confusion_matrix()
        self.assertTrue((conf==T.Tensor([[3,0,1,0],[0,2,1,0],[0,0,0,0],[0,1,0,0]])).all())

    def test_tp_tn_fp_fn(self):
        # TEST ALSO AGAINST CUSTOM METRIC

        class Truthfulness(MetricConfusionMatrixBase):
            def __init__(self, num_classes, labels=None, predictions_to_classes_convertor=None, threshold=0.5):
                super(Truthfulness, self).__init__(num_classes=num_classes, labels=labels,  predictions_to_classes_convertor = predictions_to_classes_convertor, threshold=threshold)
                self.tp = TruePositives(num_classes=num_classes, threshold=threshold) # we exploit TruePositives for the computation
                self.tn = TrueNegatives(num_classes=num_classes, threshold=threshold) # we exploit TrueNegatives for the computation

            def __call__(self, y_pred, y_true):
                tp_res = self.tp(y_pred, y_true)
                tn_res = self.tn(y_pred, y_true)
                return tp_res + tn_res


        # BINARY
        labels = ["a", "b"]
        num_classes = len(labels)
        # 4 TP, 3 TN, 5 FP, 2 FN
        actual    = T.Tensor([1,1,1,1,0,0,0,0,0,0,0,0,1,1])
        predicted = T.Tensor([1,1,1,1,0,0,0,1,1,1,1,1,0,0])

        confusion_matrix = ConfusionMatrix(num_classes=num_classes, labels=labels)
        confusion_matrix.update_state(predicted, actual)

        metric = TruePositives(num_classes)
        MetricConfusionMatrixBase.confusion_matrix_ = confusion_matrix #IN REAL-TIME THIS IS BEING HANDLED BY TRAINER-STATS

        MetricConfusionMatrixBase.confusion_matrix_ = confusion_matrix #IN REAL-TIME THIS IS BEING HANDLED BY TRAINER-STATS
        result = metric(predicted, actual)
        self.assertTrue((result==T.Tensor([4])).all())


        metric = TrueNegatives(num_classes)
        MetricConfusionMatrixBase.confusion_matrix_ = confusion_matrix #IN REAL-TIME THIS IS BEING HANDLED BY TRAINER-STATS
        result = metric(predicted, actual)
        self.assertTrue((result==T.Tensor([3])).all())


        metric = FalsePositives(num_classes)
        MetricConfusionMatrixBase.confusion_matrix_ = confusion_matrix #IN REAL-TIME THIS IS BEING HANDLED BY TRAINER-STATS
        result = metric(predicted, actual)
        self.assertTrue((result==T.Tensor([5])).all())


        metric = FalseNegatives(num_classes)
        MetricConfusionMatrixBase.confusion_matrix_ = confusion_matrix #IN REAL-TIME THIS IS BEING HANDLED BY TRAINER-STATS
        result = metric(predicted, actual)
        self.assertTrue((result==T.Tensor([2])).all())


        metric = Truthfulness(num_classes)
        MetricConfusionMatrixBase.confusion_matrix_ = confusion_matrix #IN REAL-TIME THIS IS BEING HANDLED BY TRAINER-STATS
        result = metric(predicted, actual)
        self.assertTrue((result==T.Tensor([7])).all())


        # BINARY AS 2 CLASSES
        labels = ["a", "b"]
        num_classes = len(labels)
        emb = nn.Embedding(num_classes,num_classes)
        emb.weight.data = T.eye(num_classes)
        # 4 TP, 3 TN, 5 FP, 2 FN
        actual    = T.Tensor([1,1,1,1,0,0,0,0,0,0,0,0,1,1])
        predicted = T.Tensor([1,1,1,1,0,0,0,1,1,1,1,1,0,0]).long()
        predicted  = emb(predicted)

        confusion_matrix = ConfusionMatrix(num_classes=num_classes, labels=labels)
        confusion_matrix.update_state(predicted, actual)

        metric = TruePositives(num_classes)
        MetricConfusionMatrixBase.confusion_matrix_ = confusion_matrix #IN REAL-TIME THIS IS BEING HANDLED BY TRAINER-STATS
        result = metric(predicted, actual)
        self.assertTrue((result==T.Tensor([4])).all())


        metric = TrueNegatives(num_classes)
        MetricConfusionMatrixBase.confusion_matrix_ = confusion_matrix #IN REAL-TIME THIS IS BEING HANDLED BY TRAINER-STATS
        result = metric(predicted, actual)
        self.assertTrue((result==T.Tensor([3])).all())


        metric = FalsePositives(num_classes)
        MetricConfusionMatrixBase.confusion_matrix_ = confusion_matrix #IN REAL-TIME THIS IS BEING HANDLED BY TRAINER-STATS
        result = metric(predicted, actual)
        self.assertTrue((result==T.Tensor([5])).all())


        metric = FalseNegatives(num_classes)
        MetricConfusionMatrixBase.confusion_matrix_ = confusion_matrix #IN REAL-TIME THIS IS BEING HANDLED BY TRAINER-STATS
        result = metric(predicted, actual)
        self.assertTrue((result==T.Tensor([2])).all())


        metric = Truthfulness(num_classes)
        MetricConfusionMatrixBase.confusion_matrix_ = confusion_matrix #IN REAL-TIME THIS IS BEING HANDLED BY TRAINER-STATS
        result = metric(predicted, actual)
        self.assertTrue((result==T.Tensor([7])).all())


        # 4 classes

        labels = ["a", "b", "c", "d"]
        num_classes = len(labels)
        emb = nn.Embedding(num_classes,num_classes)
        emb.weight.data = T.eye(num_classes)
        actual    = T.Tensor([0,0,0,2,2,1,1,1])
        predicted = T.Tensor([0,0,0,0,1,1,1,3]).long()
        predicted  = emb(predicted)

        confusion_matrix = ConfusionMatrix(num_classes, labels=labels)
        confusion_matrix.update_state(predicted, actual)

        # [[3,0,1,0],
        #  [0,2,1,0],
        #  [0,0,0,0],
        #  [0,1,0,0]]

        metric = TruePositives(num_classes)
        MetricConfusionMatrixBase.confusion_matrix_ = confusion_matrix #IN REAL-TIME THIS IS BEING HANDLED BY TRAINER-STATS
        result = metric(predicted, actual)
        self.assertTrue((result==T.Tensor([3,2,0,0])).all())


        metric = TrueNegatives(num_classes)
        MetricConfusionMatrixBase.confusion_matrix_ = confusion_matrix #IN REAL-TIME THIS IS BEING HANDLED BY TRAINER-STATS
        result = metric(predicted, actual)
        self.assertTrue((result==T.Tensor([4,4,6,7])).all())


        metric = FalsePositives(num_classes)
        MetricConfusionMatrixBase.confusion_matrix_ = confusion_matrix #IN REAL-TIME THIS IS BEING HANDLED BY TRAINER-STATS
        result = metric(predicted, actual)
        self.assertTrue((result==T.Tensor([1,1,0,1])).all())


        metric = FalseNegatives(num_classes)
        MetricConfusionMatrixBase.confusion_matrix_ = confusion_matrix #IN REAL-TIME THIS IS BEING HANDLED BY TRAINER-STATS
        result = metric(predicted, actual)
        self.assertTrue((result==T.Tensor([0,1,2,0])).all())


        metric = Truthfulness(num_classes)
        MetricConfusionMatrixBase.confusion_matrix_ = confusion_matrix #IN REAL-TIME THIS IS BEING HANDLED BY TRAINER-STATS
        result = metric(predicted, actual)
        self.assertTrue((result==T.Tensor([7,6,6,7])).all())


        # 4 CLASSES MULTI-LABEL

        labels = ["a", "b", "c", "d"]
        num_classes = len(labels)
        emb = nn.Embedding(num_classes,num_classes)
        emb.weight.data = T.Tensor([[1,0,1,1],[0,1,0,0],[0,0,1,1],[0,0,0,1]])
        actual    = T.Tensor([0,0,0,2,2,1,1,1])
        predicted = T.Tensor([0,0,0,0,1,1,1,3]).long()
        predicted  = emb(predicted)

        def multilabel_ypred_to_indices(y_pred, t_true):
            # FOR SIMPLICITY, THE FIRST "ARGMAX" INDEX FOUND, IS CONSIDERED THE CHOSEN CLASS
            classes = T.max(y_pred, dim=1)[1]
            return classes

        confusion_matrix = ConfusionMatrix(num_classes, labels=labels, predictions_to_classes_convertor=multilabel_ypred_to_indices)
        confusion_matrix.update_state(predicted, actual)

        metric = TruePositives(num_classes)
        MetricConfusionMatrixBase.confusion_matrix_ = confusion_matrix #IN REAL-TIME THIS IS BEING HANDLED BY TRAINER-STATS
        result = metric(predicted, actual)
        self.assertTrue((result==T.Tensor([3,2,0,0])).all())


        metric = TrueNegatives(num_classes)
        MetricConfusionMatrixBase.confusion_matrix_ = confusion_matrix #IN REAL-TIME THIS IS BEING HANDLED BY TRAINER-STATS
        result = metric(predicted, actual)
        self.assertTrue((result==T.Tensor([4,4,6,7])).all())


        metric = FalsePositives(num_classes)
        MetricConfusionMatrixBase.confusion_matrix_ = confusion_matrix #IN REAL-TIME THIS IS BEING HANDLED BY TRAINER-STATS
        result = metric(predicted, actual)
        self.assertTrue((result==T.Tensor([1,1,0,1])).all())


        metric = FalseNegatives(num_classes)
        MetricConfusionMatrixBase.confusion_matrix_ = confusion_matrix #IN REAL-TIME THIS IS BEING HANDLED BY TRAINER-STATS
        result = metric(predicted, actual)
        self.assertTrue((result==T.Tensor([0,1,2,0])).all())


        metric = Truthfulness(num_classes)
        MetricConfusionMatrixBase.confusion_matrix_ = confusion_matrix #IN REAL-TIME THIS IS BEING HANDLED BY TRAINER-STATS
        result = metric(predicted, actual)
        self.assertTrue((result==T.Tensor([7,6,6,7])).all())

