import torch.nn as nn
from dataset import CheXpert
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import libauc


class Trainer:
    def __init__(self,
                 model: nn.Module,
                 train_set: CheXpert,
                 valid_set: CheXpert,
                 criterion: str,
                 optimizer: str,
                 learning_rate: float,
                 epochs: int,
                 lr_scheduler: list[str] or str,
                 write_summary: bool,
                 early_stopping: bool,
                 early_stopping_patience: int = 5,
                 plateau_patience: int = 3,
                 exponential_gamma: float = 0.01,
                 cyclic_lr: tuple[float, float] = (0.001, 0.01)
                 ):

        # Basic necessities
        self.model = model
        self.trainSet = train_set
        self.validSet = valid_set
        self.lossFunction = None
        self.optimizer = None
        self.lr_schedule = []
        self.learningRate = learning_rate
        self.epochs = epochs

        # string and bool selectors
        self.optimizerFunction = optimizer
        self.criterion = criterion
        self.write_summary = write_summary
        self.lr_scheduler_list = lr_scheduler
        self.early_stopping = early_stopping

        # various hyperparameters
        self.early_stopping_patience = early_stopping_patience
        self.plateau_patience = plateau_patience
        self.exponential_gamma = exponential_gamma
        self.cyclic_lr = cyclic_lr

    def set_optimizer(self):
        optimizer = self.optimizerFunction.lower()

        optimizers = {
            "adadelta": optim.adadelta,
            "adagrad": optim.adagrad,
            "adam": optim.Adam,
            "adamw": optim.AdamW,
            "adamax": optim.Adamax,
            "sgd": optim.sgd,
            "asgd": optim.asgd,
            "nadam": optim.nadam,
            "radam": optim.RAdam,
            "rmsprop": optim.rmsprop,
            "rprop": optim.rprop,
            "lbfgs": optim.lbfgs
        }

        assert optimizer in optimizers.keys(), "Invalid optimizer!"
        self.optimizer = optimizers[optimizer](self.model.parameters(), lr=self.learningRate)

    def set_criterion(self):
        criterion = self.criterion

        losses = {
            "ce": nn.CrossEntropyLoss(),
            "bce": nn.BCELoss(),
            "hinge": nn.HingeEmbeddingLoss()
        }

        assert criterion in losses.keys(), "Invalid loss function!"

        self.lossFunction = losses[criterion]

    def set_criterion_libauc(self):
        # TODO
        pass

    def early_stopping(self):
        # TODO
        pass

    def train(self):

        self.set_criterion()
        self.set_optimizer()

        for epoch in range(self.epochs):
            pass

    def validate(self):
        # TODO
        pass

    def save_model(self):
        # TODO
        pass


