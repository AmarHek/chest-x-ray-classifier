from typing import Union

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score


def tensor_to_numpy(tensor: Union[torch.Tensor, np.array]):
    if type(tensor) == torch.Tensor:
        return tensor.to("cpu").numpy()
    else:
        return tensor


def multi_label_auroc(y_gt, y_pred, average=None):
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


metrics_selector = {
    "auroc": multi_label_auroc,
    "precision": precision,
    "recall": recall,
    "f1": f1
}


def metric_is_valid(metric: str):
    if metric in metrics_selector.keys():
        return True
    else:
        print("WARNING: Metric %s not in list of metrics, skipping." % metric)
        return False
