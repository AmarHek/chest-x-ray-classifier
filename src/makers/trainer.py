import numpy as np
import torch.cuda
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import os
from tqdm import tqdm

from components import get_optimizer
from datasets import load_dataset
from components import multi_label_auroc, get_loss,
from params import TrainParams, ModelParams, DatasetParams, AugmentationParams
from models import load_model


class Trainer:

    def __init__(self,
                 trainParams: TrainParams,
                 modelParams: ModelParams,
                 dataTrainParams: DatasetParams,
                 dataValParams: DatasetParams,
                 augmentationParams: AugmentationParams
                 ):

        # track params
        self.trainParams = trainParams
        self.modelParams = modelParams
        self.dataTrainParams = dataTrainParams
        self.dataValParams = dataValParams
        self.augmentationParams = augmentationParams

        # various variable declarations
        # TODO: implement device loader
        self.device = self.get_device(trainParams.device)
        # TODO: implement logger loader
        self.writer = None

        # Model Loading
        self.model = load_model(modelParams)
        train_set = load_dataset(dataTrainParams, augmentationParams)
        valid_set = load_dataset(dataValParams, augmentationParams)
        self.trainLoader, self.validLoader = self.set_dataloaders(train_set, valid_set)
        self.batch_size = trainParams.batch_size

        self.current_epoch = 0
        self.set_device()

        self.n_epochs = trainParams.n_epochs
        self.learning_rate = trainParams.learning_rate
        self.seed = trainParams.seed
        self.loss = get_loss(**trainParams.to_dict())
        self.optimizer = get_optimizer(trainParams.optimizer,
                                       learning_rate=self.learning_rate,
                                       model=self.model,
                                       loss=self.loss,
                                       **self.trainParams.to_dict())

        # saving and logging
        self.update_steps = trainParams.update_steps
        self.save_epoch_freq = trainParams.save_epoch_freq
        self.max_keep_ckpts = trainParams.max_keep_ckpts
        self.write_summary = write_logs

        # validation
        self.validation_metric = trainParams.validation_metric
        self.validation_metric_mode = trainParams.validation_metric_mode
        self.best_score = 0
        self.current_score = 0

        # early stopping
        self.early_stopping = trainParams.early_stopping
        self.early_stopping_patience = trainParams.early_stopping_patience
        self.early_stopping_tracker = 0

        # learning rate schedules
        # TODO check mode max/min
        self.lr_scheduler = get_scheduler(trainParams.lr_policy, self.optimizer, **trainParams.to_dict())

    def set_device(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device set to {self.device}.")

    # TODO: implement dataloader loader
    def set_dataloaders(self, batch_size=32, num_workers=2):
        print("Setting up dataloaders")
        train_loader = DataLoader(self.train_set, batch_size=batch_size,
                                  num_workers=num_workers, drop_last=True, shuffle=True)
        valid_loader = DataLoader(self.valid_set, batch_size=batch_size,
                                  num_workers=num_workers, drop_last=False, shuffle=False)

        return train_loader, valid_loader

    def set_epochs(self, new_epochs: int):
        self.n_epochs = new_epochs

    def set_summary_writer(self, location: str = None, comment: str = ""):
        self.writer = SummaryWriter(log_dir=location, comment=comment)

    def check_early_stopping(self, improvement: bool) -> bool:
        if improvement:
            print("Improvement detected, resetting early stopping patience.")
            self.early_stopping_tracker = 0
        else:
            self.early_stopping_tracker += 1
            print(f"No improvement. Incrementing Early Stopping tracker to {self.early_stopping_tracker}")

        return self.early_stopping_tracker > self.early_stopping_patience

    def validation_improvement(self) -> bool:
        if self.validation_metric_mode == "max":
            condition = (self.best_score < self.current_score)
        elif self.validation_metric_mode == "min":
            condition = (self.best_score > self.current_score)
        else:
            raise ValueError("Invalid validation metric mode!")

        return condition

    def train_epoch(self) -> float:
        # training for a single epoch

        # switch model to training mode
        self.model.train()
        training_loss = 0

        print(f'Training at epoch {self.current_epoch}...')

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
            if (batch + 1) % self.update_steps == 0:
                print(f'[{self.current_epoch + 1}, {batch + 1:5d}] loss: {training_loss / (batch+1):.3f}')
                if self.write_summary and self.writer is not None:
                    self.writer.add_scalar('Loss/train', training_loss/(batch+1), (self.current_epoch+1)*(batch+1))

        # clear memory
        del images, labels, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # return average loss of training set
        return training_loss/len(self.train_loader)

    def validate(self) -> (float, float):

        print(f'Validating at epoch {self.current_epoch}...')

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
                valid_loss += loss.item()
        auc = multi_label_auroc(ground_truth, predictions, average='micro')
        print(f'Validation loss at epoch {self.current_epoch+1}: {valid_loss/len(self.valid_loader)}')
        print(f'Micro-averaged AUC at epoch {self.current_epoch+1}: {auc}')

        if self.write_summary and self.writer is not None:
            self.writer.add_scalar('Loss/val', valid_loss/len(self.valid_loader),
                                   (self.current_epoch+1)*len(self.train_loader))
            self.writer.add_scalar('AUC/val', auc, (self.current_epoch+1)*len(self.train_loader))

        # Clear memory
        del images, labels, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # return average loss of val set and average auroc score
        return valid_loss/len(self.valid_loader), auc

    def train(self, base_path: str, model_name_base: str, log_path_name: str = "logs"):

        # set up path
        experiment_path = os.path.join(base_path, model_name_base)

        # set seed
        torch.manual_seed(self.seed)

        print(f'Starting training at path {experiment_path}.')

        # create experiment_path
        if not os.path.isdir(experiment_path):
            print(f"{experiment_path} does not exist. Creating directory.")
            os.mkdir(experiment_path)

        # init best score depending on score criterion
        if self.use_auc_on_val:
            print("AUC picked as val improvement score.")
            mode = "max"
            self.best_score = 0
            self.current_score = 0
        else:
            print("Loss picked as val improvement score.")
            mode = "min"
            self.best_score = np.infty
            self.current_score = np.infty

        if self.lr_scheduler is not None:
            print(f"Setting up scheduler {self.lr_scheduler}")
            self.set_lr_scheduler(mode=mode)

        # set model to device
        self.model.to(self.device)

        # set up writer
        if self.write_summary:
            log_path = os.path.join(experiment_path, log_path_name)
            print(f"Summary Writer enabled, setting up with log_path {log_path}")
            self.set_summary_writer(location=log_path)

        for epoch in tqdm(range(self.n_epochs)):

            self.current_epoch = epoch

            # Training
            self.train_epoch()

            # Validation
            val_loss, val_auc = self.validate()

            if self.use_auc_on_val:
                self.current_score = val_auc
            else:
                self.current_score = val_loss

            # update learning rate
            if self.lr_scheduler == "plateau":
                self.scheduler.step(self.current_score)
            elif self.scheduler is not None:
                self.scheduler.step()

            # save model on epoch
            if self.save_on_epoch:
                self.save_model_dict(os.path.join(experiment_path, model_name_base + f"_epoch_{epoch}.pth"))

            # save best model
            improvement = self.validation_improvement()
            if improvement:
                print(f'Model improved at epoch {epoch}, saving model.')
                self.best_score = self.current_score
                self.save_model_dict(os.path.join(experiment_path, model_name_base + f"_{epoch}_best.pth"))
                self.save_model_full(os.path.join(experiment_path, model_name_base + "_best.pt"))

            # early stopping
            if self.early_stopping:
                if self.check_early_stopping(improvement):
                    print(f"Early stopping at {epoch}")
                    break

    def save_model_dict(self, model_path: str):
        torch.save({"model": self.model.state_dict(),
                    "modelParams": self.modelParams,
                    "optimizer": self.optimizer.state_dict(),
                    "epoch": self.current_epoch,
                    "validation_metric": self.validation_metric,
                    "validation_metric_mode": self.validation_metric_mode,
                    "score": self.current_score,
                    "best_score": self.best_score}, model_path)

    def save_model_full(self, model_path: str):
        torch.save(self.model, model_path)

    def load_model_dict(self, load_path: str):
        load_dict = torch.load(load_path)
        self.modelParams = load_dict["modelParams"]
        self.model = load_model(self.modelParams)
        self.optimizer.load_state_dict(load_dict["optimizer"])
        self.current_epoch = load_dict["epoch"]
        self.best_score = load_dict["best_score"]
        self.current_score = load_dict["score"]
        self.validation_metric = load_dict["validation_metric"]
        self.validation_metric_mode = load_dict["validation_metric_mode"]
