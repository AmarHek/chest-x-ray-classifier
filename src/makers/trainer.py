from datetime import datetime

import numpy as np
import torch.cuda
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import os
from tqdm import tqdm

from components import get_optimizer, get_loss, get_scheduler, load_metrics
from datasets import load_dataset
from params import TrainParams, DatasetParams, AugmentationParams
from models import load_model


class Trainer:

    def __init__(self,
                 trainParams: TrainParams,
                 modelParams,
                 dataTrainParams,
                 dataValParams,
                 augmentParams: AugmentationParams
                 ):

        # some assertions
        assert dataTrainParams.train_labels == dataValParams.train_labels\
            , "Train and validation labels do not match!"
        assert modelParams.num_classes == len(dataTrainParams.train_labels)\
            , "Number of classes does not match number of labels!"

        # track params
        self.trainParams = trainParams
        self.modelParams = modelParams
        self.dataTrainParams = dataTrainParams
        self.dataValParams = dataValParams
        self.augmentParams = augmentParams

        # get unique work dir and initialize it
        # add exp_name to work_dir
        self.work_dir = os.path.join(self.trainParams.work_dir, self.trainParams.exp_name)
        # add current date and time to work_dir
        self.work_dir = self.work_dir + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # update experiment name
        self.trainParams.exp_name = os.path.basename(self.work_dir)
        # create the work_dir
        os.makedirs(self.work_dir)

        # seed
        self.seed = trainParams.seed

        # various variable declarations
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device set to {self.device}.")

        # Model Loading
        self.model = load_model(modelParams)

        # TODO: auslagern
        # change resolution in dataParams for some pretrained models (currently only ViT)
        if modelParams.name == "ViT":
            self.dataTrainParams.image_size = (384, 384)
            self.dataValParams.image_size = (384, 384)

        # Data Loading
        self.train_set = load_dataset(dataTrainParams, augmentParams)
        self.valid_set = load_dataset(dataValParams, augmentParams)
        self.train_loader, self.val_loader = self.set_dataloaders(trainParams.batch_size, trainParams.num_workers)

        # epoch tracking
        self.current_epoch = 0
        self.n_epochs = trainParams.n_epochs

        # losses, metrics, optimizer, etc.
        self.learning_rate = trainParams.learning_rate
        self.loss = get_loss(**trainParams.to_dict())
        self.optimizer = get_optimizer(model=self.model,
                                       loss_fn=self.loss,
                                       **self.trainParams.to_dict())
        self.metrics = load_metrics(trainParams.metrics,
                                    num_classes=self.modelParams.num_classes,
                                    threshold=trainParams.threshold,
                                    task='multilabel')
        self.train_scores = {'loss': 0,
                             **{metric: 0 for metric in trainParams.metrics}}
        self.val_scores = {'loss': 0,
                           **{metric: 0 for metric in trainParams.metrics}}

        # saving and logging
        self.update_steps = trainParams.update_steps
        self.save_epoch_freq = trainParams.save_epoch_freq
        self.max_keep_ckpts = trainParams.max_keep_ckpts
        self.saved_checkpoints = []

        # initialize logger
        if self.trainParams.logger == "tensorboard":
            os.makedirs(os.path.join(self.work_dir, 'logs'), exist_ok=True)
            self.logger = SummaryWriter(log_dir=os.path.join(self.work_dir, 'logs'))
        else:
            self.logger = None

        # validation
        if trainParams.validation_metric != "loss":
            assert trainParams.validation_metric in trainParams.metrics, "Validation metric not in metrics!"
        self.validation_metric = trainParams.validation_metric
        self.validation_metric_mode = trainParams.validation_metric_mode

        # init best score depending on score criterion
        if trainParams.validation_metric_mode == "max":
            self.best_score = 0
            self.current_score = 0
        elif trainParams.validation_metric_mode == "min":
            self.best_score = np.infty
            self.current_score = np.infty
        else:
            raise ValueError("Invalid validation metric mode!")

        # early stopping
        self.early_stopping = trainParams.early_stopping
        self.early_stopping_patience = trainParams.early_stopping_patience
        self.early_stopping_tracker = 0

        # learning rate schedules
        self.lr_scheduler = get_scheduler(self.trainParams.lr_policy,
                                          optimizer_fn=self.optimizer,
                                          mode=self.validation_metric_mode,
                                          epoch_count=self.current_epoch,
                                          **self.trainParams.to_dict())

        if self.trainParams.continue_train:
            self.load_model()

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
        self.logger = SummaryWriter(log_dir=location, comment=comment)

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

    def log(self, mode: str = "train", batch: int = None):
        if mode == "train":
            scores = self.train_scores
            divider = (batch + 1)
            log_output = f'[Epoch {self.current_epoch + 1}, Batch {batch + 1:5d}]: '
            step = (self.current_epoch + 1) * (batch + 1)
        elif mode == "val":
            scores = self.val_scores
            divider = len(self.val_loader)
            log_output = f'Validation Scores at epoch {self.current_epoch + 1}: '
            step = (self.current_epoch + 1) * len(self.train_loader)
        else:
            raise ValueError("Invalid mode!")

        for metric in scores.keys():
            # average loss and metrics at current batch
            score = scores[metric] / divider
            log_output += f"{metric.capitalize()}: {score:.4f}" + " | "
            if self.logger is not None:
                self.logger.add_scalar(f"{metric.capitalize()}/{mode}", score, step)

        # remove final " | " before printing
        log_output = log_output[:-3]
        print(log_output)

    def train_epoch(self):
        # training for a single epoch

        # switch model to training mode
        self.model.train()
        # reset running metrics
        for metric in self.train_scores.keys():
            self.train_scores[metric] = 0

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

            # track loss after each batch
            self.train_scores["loss"] += loss.item()

            # Update metrics after each batch
            for metric in self.metrics.keys():
                # compute remaining metrics
                self.train_scores[metric] += self.metrics[metric](pred, labels.int())

            # Logging
            if (batch + 1) % self.update_steps == 0:
                self.log(mode="train", batch=batch)

        # clear memory
        del images, labels, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def validate(self):

        print(f'Validating at epoch {self.current_epoch}...')

        # Put model in eval mode
        self.model.eval()

        # reset loss and metrics
        for metric in self.val_scores.keys():
            self.val_scores[metric] = 0

        # tensors to collect predictions and ground truths
        predictions = torch.FloatTensor().to(self.device)
        ground_truth = torch.FloatTensor().to(self.device)

        with torch.no_grad():
            for batch, (images, labels) in enumerate(self.val_loader):
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
                self.val_scores['loss'] += loss.item()

        # average loss
        self.val_scores['loss'] /= len(self.val_loader)
        # update metrics
        for metric in self.metrics.keys():
            # compute remaining metrics
            self.val_scores[metric] = self.metrics[metric](predictions, ground_truth.int())

        # Logging
        self.log(mode="val")

        # Clear memory
        del images, labels, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def train(self):
        # set seed
        torch.manual_seed(self.seed)

        print(f'Starting training at path {self.work_dir}.')

        # set model to device
        self.model.to(self.device)

        for epoch in tqdm(range(self.n_epochs)):
            self.current_epoch = epoch

            # Training
            self.train_epoch()

            # Validation
            self.validate()
            self.current_score = self.val_scores[self.validation_metric]

            # update learning rate
            if self.trainParams.lr_policy == "plateau":
                self.lr_scheduler.step(self.current_score)
            elif self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # save model on epoch
            if (self.current_epoch % self.save_epoch_freq) == 0:
                self.save_model(save_best=False)

            # save best model
            improvement = self.validation_improvement()
            if improvement:
                print(f'Model improved at epoch {epoch}, saving model.')
                self.best_score = self.current_score
                self.save_model(save_best=True)

            # early stopping
            if self.early_stopping:
                if self.check_early_stopping(improvement):
                    print(f"Early stopping at {epoch}")
                    break

    def print_params(self):
        if self.trainParams.continue_train:
            print(f"Continuing training from epoch {self.current_epoch}...")
        else:
            print(f"Starting training from epoch 0...")

        print(f"Experiment name: {self.trainParams.exp_name}")
        print(f"Work directory: {self.work_dir}")
        self.trainParams.print_params()
        self.modelParams.print_params()
        self.dataTrainParams.print_params()
        self.dataValParams.print_params()
        self.augmentParams.print_params()

    def save_params(self, addTestParams: bool = True,
                    testDataParams: DatasetParams = None):
        """
        Saves all parameters in the work_dir
        Args:
            addTestParams: If true, creates entry testParams for easier testing
            testDataParams (optional): Can be passed to automatically add parameters for test dataset

        Returns:

        """

        # convert params to dicts
        params = {'trainParams': self.trainParams.to_dict(),
                  'modelParams': self.modelParams.to_dict(),
                  'dataTrainParams': self.dataTrainParams.to_dict(),
                  'dataValParams': self.dataValParams.to_dict(),
                  'augmentParams': self.augmentParams.to_dict()}

        if addTestParams:
            testParams = {'exp_name': self.trainParams.exp_name,
                          'work_dir': self.trainParams.work_dir,
                          'metrics': self.trainParams.metrics}
            params['testParams'] = testParams

        if testDataParams is not None:
            params['dataTestParams'] = testDataParams.to_dict()

        with open(os.path.join(self.work_dir, 'params.yaml'), 'w') as f:
            yaml.dump(params, f)

    def save_model(self, save_best: bool = False):
        if not save_best:
            filename = f"{self.trainParams.exp_name}_{self.current_epoch}.pth"

            # append filename to tracker and remove old checkpoints
            self.saved_checkpoints.append(filename)
            if len(self.saved_checkpoints) > self.trainParams.max_keep_ckpts:
                oldest_checkpoint = self.saved_checkpoints.pop(0)
                os.remove(os.path.join(self.work_dir, oldest_checkpoint))
        else:
            filename = f"{self.trainParams.exp_name}_{self.current_epoch}_best.pth"
            # remove previous best checkpoint
            for f in os.listdir(self.work_dir):
                if f.endswith('.pth') and 'best' in f:
                    os.remove(os.path.join(self.work_dir, f))

        save_path = os.path.join(self.work_dir, filename)

        torch.save({"model": self.model.state_dict(),
                    "modelParams": self.modelParams,
                    "optimizer": self.optimizer.state_dict(),
                    "epoch": self.current_epoch,
                    "validation_metric": self.validation_metric,
                    "validation_metric_mode": self.validation_metric_mode,
                    "score": self.current_score,
                    "best_score": self.best_score}, save_path)

    def load_model(self):
        # find the latest checkpoint in the work_dir that does not contain 'best' in the name
        checkpoints = [f for f in os.listdir(self.trainParams.work_dir) if f.endswith('.pth') and 'best' not in f]
        if not checkpoints:
            raise FileNotFoundError("No checkpoint found in %s" % self.trainParams.work_dir)

        checkpoints.sort()
        load_path = os.path.join(self.trainParams.work_dir, checkpoints[-1])

        loaded_dict = torch.load(load_path)

        self.modelParams = loaded_dict["modelParams"]
        self.model = load_model(self.modelParams)
        self.optimizer.load_state_dict(loaded_dict["optimizer"])
        self.current_epoch = loaded_dict["epoch"]
        self.best_score = loaded_dict["best_score"]
        self.current_score = loaded_dict["score"]
        self.validation_metric = loaded_dict["validation_metric"]
        self.validation_metric_mode = loaded_dict["validation_metric_mode"]
