import numpy as np
import torch.cuda
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import libauc
import os
from tqdm import tqdm

from metrics import multi_label_auroc


class Trainer:

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
        "lbfgs": optim.LBFGS
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
                 batch_size: int = 32,
                 update_steps: int = 2000,
                 save_on_epoch: bool = True,
                 use_auc_on_val: bool = False,
                 early_stopping: bool = True,
                 early_stopping_patience: int = 5,
                 lr_scheduler: str = None,
                 plateau_patience: int = 3,
                 exponential_gamma: float = 0.01,
                 cyclic_lr: tuple[float, float] = (0.001, 0.01),
                 write_summary: bool = True,
                 ):

        # various variable declarations
        self.train_loader = None
        self.valid_loader = None
        self.loss = None
        self.optimizer = None
        self.lr_scheduler = None
        self.device = None
        self.writer = None

        # Basic necessities
        self.model = model
        self.train_set = train_set
        self.valid_set = valid_set
        self.set_dataloaders(batch_size)
        self.update_steps = update_steps
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.set_device()
        self.set_loss(loss)
        self.set_optimizer(optimizer, learning_rate)

        # string and bool selectors
        self.save_on_epoch = save_on_epoch
        self.use_auc_on_val = use_auc_on_val
        self.write_summary = write_summary

        # early stopping
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_tracker = 0

        # learning rate schedules
        self.plateau_patience = plateau_patience
        self.exponential_gamma = exponential_gamma
        self.cyclic_lr = cyclic_lr
        if lr_scheduler is not None:
            self.set_lr_scheduler(lr_scheduler)

    def set_device(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def set_dataloaders(self, batch_size=32, num_workers=2):
        self.train_loader = DataLoader(self.train_set, batch_size=batch_size,
                                       num_workers=num_workers, drop_last=True, shuffle=True)
        self.valid_loader = DataLoader(self.valid_set, batch_size=batch_size,
                                       num_workers=num_workers, drop_last=False, shuffle=False)

    def set_epochs(self, new_epochs: int):
        self.epochs = new_epochs

    def set_loss(self, loss):
        loss = loss.lower()
        assert loss in Trainer.losses.keys(), "Invalid loss!"

        self.loss = Trainer.losses[loss]

    def set_optimizer(self, optimizer, learning_rate):
        optimizer = optimizer.lower()
        assert optimizer in Trainer.optimizers.keys(), "Invalid optimizer!"
        self.optimizer = Trainer.optimizers[optimizer](self.model.parameters(), lr=learning_rate)

    def set_lr_scheduler(self, lr_scheduler):
        assert self.optimizer is not None, "Optimizer Function needs to be set before the scheduler!"

        lr_scheduler = lr_scheduler.lower()
        if lr_scheduler == "plateau":
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                           patience=self.plateau_patience)
        elif lr_scheduler == "exponential_decay":
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                                       gamma=self.exponential_gamma)
        elif lr_scheduler == "cyclic":
            self.lr_scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer,
                                                                  base_lr=self.cyclic_lr[0],
                                                                  max_lr=self.cyclic_lr[1])
        else:
            self.lr_scheduler = None
            "Invalid lr_scheduler specified!"

    def set_summary_writer(self, location: str = None, comment: str = ""):
        self.writer = SummaryWriter(log_dir=location, comment=comment)

    def check_early_stopping(self, improvement: bool) -> bool:
        if improvement:
            self.early_stopping_tracker = 0
        else:
            self.early_stopping_tracker += 1

        return self.early_stopping_tracker > self.early_stopping_patience

    def train_epoch(self, epoch) -> float:
        # training for a single epoch

        # switch model to training mode
        self.model.train()
        training_loss = 0

        print(f'Training at epoch {epoch}...')

        for batch, (images, labels) in enumerate(self.train_loader):
            # move to device
            images = images.to(self.device)
            labels = labels.to(self.device)

            # zero the gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            pred = self.model(images)
            loss = self.loss(pred, labels)

            # backprop
            loss.backward()
            self.optimizer.step()

            # Update training loss after each batch
            training_loss += loss.item()
            if batch % (self.update_steps - 1) == 0:
                print(f'[{epoch + 1}, {batch + 1:5d}] loss: {training_loss / batch:.3f}')
                if self.write_summary and self.writer is not None:
                    self.writer.add_scalars('Loss/train', training_loss/(batch+1), (epoch+1)*(batch+1))

        # clear memory
        del images, labels, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # return average loss of training set
        return training_loss/len(self.train_loader)

    def validate(self, epoch) -> (float, float):

        print(f'Validating at epoch {epoch}...')

        # Put model in eval mode
        self.model.eval()

        # running loss
        valid_loss = 0
        # tensors to collect predictions and ground truths
        predictions = torch.FloatTensor().to(self.device)
        ground_truth = torch.FloatTensor().to(self.device)

        with torch.no_grad():
            for batch, (images, labels) in enumerate(self.valid_loader):
                # move inputs to device
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward-pass
                output = self.model(images)
                loss = self.loss(output, labels)

                # update containers
                ground_truth = torch.cat((ground_truth, labels), 0)
                predictions = torch.cat((predictions, output), 0)

                # update validation loss after each batch
                valid_loss += loss
        auc = multi_label_auroc(ground_truth, predictions, average='micro')
        print(f'Validation loss at epoch {epoch+1}: {valid_loss/len(self.valid_loader)}')
        print(f'Micro-averaged AUC at epoch {epoch+1}: {auc}')

        if self.write_summary and self.writer is not None:
            self.writer.add_scalars('Loss/val', valid_loss/len(self.valid_loader), (epoch+1)*len(self.train_loader))
            self.writer.add_scalars('AUC/val', auc, (epoch+1)*len(self.train_loader))

        # Clear memory
        del images, labels, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # return average loss of val set and average auroc score
        return valid_loss/len(self.valid_loader), auc

    def train(self, base_path: str, model_name_base: str, log_path_name: str = "logs"):

        experiment_path = os.path.join(base_path, model_name_base)

        # create experiment_path
        if not os.path.isdir(experiment_path):
            os.mkdir(experiment_path)

        # init best score depending on score criterion
        if self.use_auc_on_val:
            best_score = 0
        else:
            best_score = np.infty

        # set model to device
        self.model.to(self.device)

        # set up writer
        if self.write_summary:
            log_path = os.path.join(experiment_path, log_path_name)
            self.set_summary_writer(location=log_path)

        for epoch in range(self.epochs):

            # Training
            train_loss = self.train_epoch(epoch)

            # Validation
            val_loss, val_auc = self.validate(epoch)

            if self.use_auc_on_val:
                new_score = val_auc
            else:
                new_score = val_loss

            # # write summary
            # if self.write_summary and self.writer is not None:
            #     self.writer.add_scalars('Loss/train', train_loss, epoch+1)
            #     self.writer.add_scalars('Loss/val', val_loss, epoch+1)
            #     self.writer.add_scalars('AUC/val', val_auc, epoch+1)

            # update learning rate
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # save model on epoch
            if self.save_on_epoch:
                self.save_model(os.path.join(base_path, model_name_base + f"_epoch_{epoch}"), epoch, new_score)

            # save best model
            if self.use_auc_on_val:
                condition = (best_score < new_score)
            else:
                condition = (best_score > new_score)
            if condition:
                best_score = new_score
                self.save_model(os.path.join(base_path, model_name_base + "_best"), epoch, best_score)

            # early stopping
            if self.early_stopping:
                if self.check_early_stopping(condition):
                    print(f"Early stopping at {epoch}")
                    break

    def save_model(self, model_path: str, epoch: int, score: float):
        if self.use_auc_on_val:
            score_name = "Val_AUC"
        else:
            score_name = "Val_Loss"
        self.model.save({"model": self.model.state_dict(),
                         "optimizer": self.optimizer.state_dict(),
                         "epoch": epoch,
                         "score_name": score_name,
                         "score": score}, model_path)
