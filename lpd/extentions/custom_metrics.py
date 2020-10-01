import torch as T

def binary_accuracy(y_pred, y_true):
    pred = y_pred >= 0.5
    truth = y_true >= 0.5
    acc = pred.eq(truth).sum().float() / y_true.numel()
    return acc

def binary_accuracy_with_logits(y_pred, y_true):
    assert y_true.ndim == 1 and y_true.size() == y_pred.size()
    y_pred_act = T.sigmoid(y_pred)
    pred = y_pred_act >= 0.5
    truth = y_true >= 0.5
    acc = pred.eq(truth).sum().float() / pred.numel()
    return acc