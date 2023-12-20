from typing import List, Union

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import torchmetrics as tm


def load_metrics(metrics: List[str],
                 num_classes: int,
                 task,
                 threshold: float = 0.5):
    metrics_selector = {
        "auc": tm.AUROC(num_labels=num_classes, task=task, average='macro'),
        "auc_class": tm.AUROC(num_labels=num_classes, task=task, average=None),
        "prec": tm.Precision(num_labels=num_classes, task=task, threshold=threshold,
                             average='macro'),
        "prec_class": tm.Precision(num_labels=num_classes, task=task, threshold=threshold,
                                   average=None),
        "rec": tm.Recall(num_labels=num_classes, task=task, threshold=threshold,
                         average='macro'),
        "rec_class": tm.Recall(num_labels=num_classes, task=task, threshold=threshold,
                               average=None),
        "f1": tm.F1Score(num_labels=num_classes, task=task, threshold=threshold,
                         average='macro'),
        "f1_class": tm.F1Score(num_labels=num_classes, task=task, threshold=threshold,
                               average=None),
    }

    metrics_dict = {}
    for metric in metrics:
        metric = metric.lower()
        if metric not in metrics_selector.keys():
            raise ValueError(f"Metric {metric} not implemented! "
                             f"Available metrics are: {metrics_selector.keys()}")
        else:
            metrics_dict[metric] = metrics_selector[metric]

    return metrics_dict


def tensor_to_numpy(tensor: Union[torch.Tensor, np.array]):
    if isinstance(tensor, torch.Tensor):
        return tensor.to("cpu").numpy()
    else:
        return tensor


def multi_label_auroc(y_gt: np.array, y_pred: np.array, average=None):
    auroc = []
    gt_np = tensor_to_numpy(y_gt)
    pred_np = tensor_to_numpy(y_pred)

    assert gt_np.shape == pred_np.shape, "y_gt and y_pred should have the same size"

    if average is None:
        for i in range(gt_np.shape[1]):
            auroc.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
        return auroc
    else:
        return roc_auc_score(gt_np, pred_np, average=average)


def precision(y_gt, y_pred, threshold=0.5, average=None):
    prec = []
    gt_np = tensor_to_numpy(y_gt)
    pred_np = tensor_to_numpy(y_pred)

    assert gt_np.shape == pred_np.shape, "y_gt and y_pred should have the same size"

    # map probabilities to hard labels
    pred_np = (pred_np > threshold).astype(int)

    if average is None:
        for i in range(gt_np.shape[1]):
            prec.append(precision_score(gt_np[:, i], pred_np[:, i], average="binary"))
        return prec
    else:
        return precision_score(gt_np, pred_np, average=average)


def recall(y_gt, y_pred, threshold=0.5, average=None):
    rec = []
    gt_np = tensor_to_numpy(y_gt)
    pred_np = tensor_to_numpy(y_pred)

    assert gt_np.shape == pred_np.shape, "y_gt and y_pred should have the same size"

    # map probabilities to hard labels
    pred_np = (pred_np > threshold).astype(int)

    if average is None:
        for i in range(gt_np.shape[1]):
            rec.append(recall_score(gt_np[:, i], pred_np[:, i], average="binary"))
        return rec
    else:
        return recall_score(gt_np, pred_np, average=average)


def f1(y_gt, y_pred, threshold=0.5, average=None):
    fone = []
    gt_np = y_gt.to("cpu").numpy()
    pred_np = y_pred.to("cpu").numpy()

    assert gt_np.shape == pred_np.shape, "y_gt and y_pred should have the same size"

    # map probabilities to hard labels
    pred_np = (pred_np > threshold).astype(int)

    if average is None:
        for i in range(gt_np.shape[1]):
            fone.append(f1_score(gt_np[:, i], pred_np[:, i], average="binary"))
        return fone
    else:
        return f1_score(gt_np, pred_np, average=average)
