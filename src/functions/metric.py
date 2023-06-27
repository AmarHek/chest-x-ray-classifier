from sklearn.metrics import roc_auc_score, precision_score, recall_score


def multi_label_auroc(y_gt, y_pred, average=None):
    auroc = []
    gt_np = y_gt.to("cpu").numpy()
    pred_np = y_pred.to("cpu").numpy()

    assert gt_np.shape == pred_np.shape, "y_gt and y_pred should have the same size"

    if average is None:
        for i in range(gt_np.shape[1]):
            auroc.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
        return auroc
    else:
        return roc_auc_score(gt_np, pred_np, average=average)


def precision(y_gt, y_pred, threshold=0.5, average=None):
    prec = []
    gt_np = y_gt.to("cpu").numpy()
    pred_np = y_pred.to("cpu").numpy()

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
    gt_np = y_gt.to("cpu").numpy()
    pred_np = y_pred.to("cpu").numpy()

    assert gt_np.shape == pred_np.shape, "y_gt and y_pred should have the same size"

    # map probabilities to hard labels
    pred_np = (pred_np > threshold).astype(int)

    if average is None:
        for i in range(gt_np.shape[1]):
            rec.append(recall_score(gt_np[:, i], pred_np[:, i], average="binary"))
        return rec
    else:
        return recall_score(gt_np, pred_np, average=average)


metrics = {
    "auroc": multi_label_auroc,
    "precision": precision,
    "recall": recall
}


def metric_is_valid(metric: str):
    if metric in metrics_dict.keys():
        return True
    else:
        print("WARNING: Metric %s not in list of metrics, skipping." % metric)
        return False
