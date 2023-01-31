import torch.nn as nn
from dataset import CheXpert
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import libauc


class Trainer:

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

    losses = {
        "ce": nn.CrossEntropyLoss(),
        "bce": nn.BCELoss(),
        "hinge": nn.HingeEmbeddingLoss()
    }

    def __init__(self,
                 model: nn.Module,
                 train_set: Dataset,
                 valid_set: Dataset,
                 loss: str,
                 optimizer: str,
                 learning_rate: float,
                 epochs: int,
                 lr_scheduler: list[str] or str,
                 early_stopping: bool,
                 early_stopping_patience: int = 5,
                 plateau_patience: int = 3,
                 exponential_gamma: float = 0.01,
                 cyclic_lr: tuple[float, float] = (0.001, 0.01),
                 write_summary: bool = True,
                 ):

        # various variable declarations
        self.train_loader = None
        self.valid_loader = None
        self.loss_function = None
        self.loss = None
        self.optimizer_function = None
        self.optimizer = None
        self.device = None
        self.writer = None

        # Basic necessities
        self.model = model
        self.train_set = train_set
        self.valid_set = valid_set
        self.lr_schedule = []
        self.learning_rate = learning_rate
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

    def set_dataloaders(self, batch_size=32, num_workers=2):
        self.trainLoader = DataLoader(self.trainSet, batch_size=batch_size,
                                      num_workers=num_workers, drop_last=True, shuffle=True)
        self.validLoader = DataLoader(self.validSet, batch_size=batch_size,
                                      num_workers=num_workers, drop_last=False, shuffle=False)

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

            running_loss = 0.0
            for i, data in enumerate(self.tra)

    def validate(self):
        # TODO
        pass

    def save_model(self):
        # TODO
        pass


