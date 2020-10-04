import torch as T

def binary_accuracy(y_pred, y_true):
    assert y_true.ndim == 1 and y_true.size() == y_pred.size()
    pred = y_pred >= 0.5
    truth = y_true >= 0.5
    acc = pred.eq(truth).sum().float() / y_true.numel()
    return acc

def binary_accuracy_with_logits(y_pred, y_true):
    y_pred_sigmoid = T.sigmoid(y_pred)
    return binary_accuracy(y_pred_sigmoid, y_true)

#NOT TESTED YET
# def categorical_accuracy(y_pred, y_true):
#     assert y_true.size() == y_pred.size()
#     indices = T.max(y_pred, 1)[1]
#     correct = T.eq(indices, y_true).view(-1)
#     return T.sum(correct).item() / correct.shape[0]

# def categorical_accuracy_with_logits(y_pred, y_true):
#     y_pred_softmax = T.softmax(y_pred)
#     return categorical_accuracy(y_pred_softmax, y_true)