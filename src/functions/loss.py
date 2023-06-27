import torch.nn as nn
from libauc.losses import APLoss, AUCMLoss, CompositionalAUCLoss, pAUCLoss


losses = {
        "ce": nn.CrossEntropyLoss(),
        "bce": nn.BCELoss(),
        "hinge": nn.HingeEmbeddingLoss(),
        "aucm": AUCMLoss(),
        "compositionalaucloss": CompositionalAUCLoss(),
        "paucloss": pAUCLoss,
        "aploss": APLoss
    }
