import torch.optim as optim
from libauc.optimizers import PESG, PDSCA, SOAP, SOPA, SOPAs, SOTAs


optimizers = {
        "adadelta": optim.Adadelta,
        "adagrad": optim.Adagrad,
        "adam": optim.Adam,
        "adamw": optim.AdamW,
        "adamax": optim.Adamax,
        "sgd": optim.SGD,
        "asgd": optim.ASGD,
        "nadam": optim.NAdam,
        "radam": optim.RAdam,
        "rmsprop": optim.RMSprop,
        "rprop": optim.Rprop,
        "lbfgs": optim.LBFGS,
        "pesg": PESG,
        "pdsca": PDSCA,
        "soap": SOAP,
        "sopa": SOPA,
        "sopas": SOPAs,
        "sotas": SOTAs
    }
