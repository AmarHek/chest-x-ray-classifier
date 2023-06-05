from sklearn.metrics import roc_auc_score


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


