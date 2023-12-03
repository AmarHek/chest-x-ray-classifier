import torch.nn as nn
from libauc.losses import APLoss, AUCMLoss, CompositionalAUCLoss, pAUCLoss


def get_loss(loss: str, **kwargs):
    loss = loss.lower()

    if loss == "ce":
        return nn.CrossEntropyLoss()
    elif loss == "bce":
        return nn.BCELoss()
    elif loss == "hinge":
        return nn.HingeEmbeddingLoss()
    elif loss == "aucm":
        return AUCMLoss()
    elif loss == "compositionalaucloss":
        return CompositionalAUCLoss(**kwargs)
    elif loss == "paucloss":
        return pAUCLoss(**kwargs)
    elif loss == "aploss":
        return APLoss(**kwargs)
    else:
        raise ValueError(f"Invalid loss function: {loss}")

