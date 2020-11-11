import torch as T
from lpd.enums.confusion_matrix_based_metric import ConfusionMatrixBasedMetric as metric

class ConfusionMatrix():
    """
        Agrs:
            num_classes - The number of classes in the classification
            labels - names of classes, for nice prints, if not provided, the class index will be the label
            predictions_to_classes_convertor - (optional) a function that takes y_pred batch and y_true batch and converts it into class indices batch where
                                               each index represents the chosen class
                                               if None:  
                                                        torch.max with indices will be used for multi-class .
                                                        threshold will be used for binary or multilabel.
            threshold - for binary or multilable classification

    """
    def __init__(self, num_classes, labels=None, predictions_to_classes_convertor=None, threshold=0.5):
        self.num_classes = num_classes
        self.class_idxs = list(range(self.num_classes))
        self.labels = self._build_labels(labels)
        self.class_idx_to_label = self._build_idx_to_label()
        self.confusion = self._build_confusion()
        self.threshold = threshold
        self.predictions_to_classes_convertor = predictions_to_classes_convertor if predictions_to_classes_convertor else self._convert_predictions_to_classes
        if self.predictions_to_classes_convertor is None:
            raise ValueError(f'[ConfusionMatrix] - Got None for predictions_to_classes_convertor, expected a function')

        self.last_stats = None

    def _build_labels(self, labels):
        if labels is None:
            return [str(c) for c in self.class_idxs]
        return labels

    def _build_idx_to_label(self):
        return dict(zip(self.class_idxs, self.labels)) 
    
    def _build_confusion(self):
        length = len(self.class_idxs)
        confusion = T.zeros((length, length)).long()
        return confusion

    def _normalized(self):
        asum = T.sum(self.confusion)
        return self.confusion / asum.float()

    def _convert_predictions_to_classes(self, y_pred: T.Tensor, y_true: T.Tensor):
        if not (len(y_pred.shape) == len(y_true.shape) or \
                len(y_pred.shape) == len(y_true.shape) + 1):
            raise ValueError("Expecting y_pred and y_true to has same amount of dimentions, or y_pred to have an extra one")

        if self.num_classes == 2:
            if len(y_pred.shape) == 2 and y_pred.shape[1] == 1:
                # BINARY AS SINGLE DIGIT WRAPPED
                classes = (y_pred.squeeze(1) >= self.threshold).long()
                return classes
            if len(y_pred.shape) == 1:
                # BINARY AS SINGLE DIGIT
                classes = (y_pred >= self.threshold).long()
                return classes

        if len(y_pred.shape) == len(y_true.shape) + 1:
            # MULTI CLASS
            classes = T.max(y_pred, dim=1)[1]
            return classes

    def get_confusion_matrix(self, normalized=False):
        if normalized:
            return self._normalized().clone().detach()
        return self.confusion.clone().detach()

    def get_stats(self):
        if self.last_stats:
            # USE CACHE
            return self.last_stats

        asum = T.sum(self.confusion)
        stats = {}
        for class_idx in self.class_idxs:
            class_stats = {}
            row_sum = T.sum(self.confusion[class_idx])
            col_sum = T.sum(self.confusion[:,class_idx])

            tp  = self.confusion[class_idx][class_idx]
            fp  = row_sum - tp
            fn  = col_sum - tp
            tn  = asum - row_sum - col_sum + tp  # +TP BECAUSE ROW_SUM AND COL_SUM REMOVES self.confusion[class_idx][class_idx] TWICE

            class_stats[metric.TP] = tp
            class_stats[metric.FP] = fp
            class_stats[metric.FN] = fn
            class_stats[metric.TN] = tn


            # SUMS
            tp_fp = (tp + fp).float()
            tp_fn = (tp + fn).float()
            tp_tn = (tp + tn).float()
            tn_fn = (tn + fn).float()
            fp_fn = (fp + fn).float()
            tn_fp = (tn + fp).float()

            class_stats[metric.PRECISION]                    = (tp / tp_fp) if tp_fp > 0 else 0.0
            class_stats[metric.SENSITIVITY]                  = (tp / tp_fn) if tp_fn > 0 else 0.0
            class_stats[metric.SPECIFICITY]                  = (tn / tn_fp) if tn_fp > 0 else 0.0
            class_stats[metric.RECALL]                       = (tp / tp_fn) if tp_fn > 0 else 0.0
            class_stats[metric.POSITIVE_PREDICTIVE_VALUE]    = (tp / tp_fp) if tp_fp > 0 else 0.0
            class_stats[metric.NEGATIVE_PREDICTIVE_VALUE]    = (tn / tn_fn) if tn_fn > 0 else 0.0
            class_stats[metric.ACCURACY]                     = (tp_tn / asum) if asum > 0 else 0.0
            class_stats[metric.F1SCORE]                      = ((2 * tp) / ((2 * tp) + fp_fn)) if ((2 * tp) + fp_fn) > 0 else 0.0


            stats[self.class_idx_to_label[class_idx]] = class_stats
        self.last_stats = stats
        return stats

    def confusion_matrix_string(self, row_prefix='\n', normalized=False):
        max_value_len = len(str(T.max(self.confusion).item()))
        cellwidth = max(7, max_value_len)

        if normalized:
            confusion = self._normalized()
        else:
            confusion = self.confusion

        def handle_confusion_row(confusion_row):
            if normalized:
                return [("%0.2f%%" % (val * 100)).ljust(cellwidth)[0:cellwidth] for val in confusion_row]
            else:
                return [str(val.item()).ljust(cellwidth)[0:cellwidth] for val in confusion_row]

        cm_strings = [f'Confusion Matrix, ᴾ = Pred, ᵀ = True', '']

        cm_strings.append("        %s" % " ".join([("%sᵀ" % label).ljust(cellwidth)[0:cellwidth] for label in self.labels]))
        for label, confusion_row in zip(self.labels, confusion):
            cm_strings.append("%s %s" % (("%sᴾ" % label).ljust(cellwidth)[0:cellwidth], " ".join(handle_confusion_row(confusion_row))))

        cm_strings.append('')
        return row_prefix.join(cm_strings)

    def reset(self):
        self.last_stats = None # INVALIDATE CACHE
        self.confusion = self.confusion * 0

    def update_state(self, y_pred: T.Tensor, y_true: T.Tensor):
        self.last_stats = None # INVALIDATE CACHE
        y_pred_class_idxs = self.predictions_to_classes_convertor(y_pred, y_true)
        y_true_class_idxs = y_true.long()

        for row, col in zip(y_pred_class_idxs, y_true_class_idxs):
            self.confusion[row][col] += 1
