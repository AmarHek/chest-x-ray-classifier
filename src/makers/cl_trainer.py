from datetime import datetime
import os
from natsort import natsorted

import numpy as np
import torch.cuda
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from components import get_optimizer, get_loss, get_scheduler, load_metrics
from datasets import load_dataset
from params import TrainParams, AugmentationParams
from models import load_model
import pandas as pd

class CLTrainer:

    def __init__(self,
                 trainParams: TrainParams,
                 modelParams,
                 dataTrainParams,
                 dataValParams,
                 augmentParams: AugmentationParams
                 ):

        # some assertions
        assert dataTrainParams.train_labels == dataValParams.train_labels, \
            "Train and validation labels do not match!"
        assert modelParams.num_classes == len(dataTrainParams.train_labels), \
            "Number of classes does not match number of labels!"

        # track params
        self.trainParams = trainParams
        self.modelParams = modelParams
        self.dataTrainParams = dataTrainParams
        self.dataValParams = dataValParams
        self.augmentParams = augmentParams

        # get unique work dir and initialize it
        if not self.trainParams.continue_train:
            # add exp_name to work_dir
            self.work_dir = os.path.join(self.trainParams.work_dir, self.trainParams.exp_name)
            # add current date and time to work_dir
            self.work_dir = self.work_dir + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            # update experiment name
            self.trainParams.exp_name = os.path.basename(self.work_dir)
            # create the work_dir
            os.makedirs(self.work_dir)
        else:
            self.work_dir = os.path.join(self.trainParams.work_dir, self.trainParams.exp_name)

        # device and seed
        self.seed = trainParams.seed
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # set model
        self.model = load_model(modelParams)

        # change resolution in dataParams for some pretrained models (currently only ViT)
        if modelParams.name == "ViT":
            self.dataTrainParams.image_size = (384, 384)
            self.dataValParams.image_size = (384, 384)

        # Data Loading
        self.train_set = load_dataset(dataTrainParams, augmentParams)
        self.valid_set = load_dataset(dataValParams, augmentParams)
        self.train_loader, self.val_loader = self.set_dataloaders(trainParams.batch_size, trainParams.num_workers,1,self.trainParams.n_epochs)

        # losses, metrics, optimizer, etc.
        self.learning_rate = trainParams.learning_rate
        self.loss = get_loss(**trainParams.to_dict())
        self.optimizer = get_optimizer(model=self.model,
                                       loss_fn=self.loss,
                                       **self.trainParams.to_dict())

        # loss and metrics trackers
        self.train_loss = 0
        self.train_metrics = load_metrics(trainParams.metrics,
                                          num_classes=self.modelParams.num_classes,
                                          threshold=trainParams.threshold,
                                          task='multilabel',
                                          device=self.device)
        self.val_loss = 0
        self.val_metrics = load_metrics(trainParams.metrics,
                                        num_classes=self.modelParams.num_classes,
                                        threshold=trainParams.threshold,
                                        task='multilabel',
                                        device=self.device)

        # logging and saving
        self.early_stopping_tracker = 0
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
        # init best score depending on score criterion
        if self.trainParams.validation_metric_mode == "max":
            self.best_score = 0
            self.current_score = 0
        elif self.trainParams.validation_metric_mode == "min":
            self.best_score = np.infty
            self.current_score = np.infty
        else:
            raise ValueError("Invalid validation metric mode!")

        # scheduling
        self.current_epoch = 0
        self.lr_scheduler = get_scheduler(self.trainParams.lr_policy,
                                          optimizer_fn=self.optimizer,
                                          mode=self.trainParams.validation_metric_mode,
                                          epoch_count=self.current_epoch,
                                          **self.trainParams.to_dict())

        if self.trainParams.continue_train:
            self.load_model()

    def set_dataloaders(self, batch_size=32, num_workers=2,current_epoch=1,n_epochs=3,cl_learning=True):
        print("Setting up CL dataloaders")
        #set training_sets
        cl_strategy = "linear"
        if cl_strategy == "linear" and cl_learning:
            max_n = len(self.train_set._images_list)
            n = int(current_epoch/n_epochs*max_n)
            print("n set at ",n)
        #get difficulties for training set
            try:
                #temporary hard coded, should be taken from trainParams
                #diff = pd.read_csv(r"C:\Users\Finn\Desktop\Informatik\4. Semester\Bachelor-Arbeit\Framework new\chest-x-ray-classifier\configs\local_train_difficulties.csv")
                print(self.dataTrainParams)
                print(self.dataTrainParams.csv_path,self.dataTrainParams.image_root_path)
                df = self.dataTrainParams.csv_path
                #image_root_path = "C:/Users/Finn/Downloads/archive (6)"
                image_root_path = self.dataTrainParams.image_root_path
                diff = pd.read(f"{df[:-4]}_difficulties.csv")
                anticurriculum = False
                if anticurriculum:
                    diff = diff.sort_values(by="difficulty",ascending=True)
                else:
                    diff = diff.sort_values(by="difficulty",ascending=False)
                
                self.train_set._images_list = [os.path.join(image_root_path, path) for path in diff['Path'].head(n).tolist()]
                self.train_set.num_images = len(self.train_set._images_list)
            except:
                print(f"No difficulties available for ..., not applying curriculum")
                n = len(self.train_set._images_list)
        
        train_loader = DataLoader(self.train_set, batch_size=batch_size,
                                  num_workers=num_workers, drop_last=True, shuffle=True)
        valid_loader = DataLoader(self.valid_set, batch_size=batch_size,
                                  num_workers=num_workers, drop_last=False, shuffle=False)
        print("Trainloader has length", len(train_loader))

        return train_loader, valid_loader

    def check_early_stopping(self, improved: bool) -> bool:
        if improved:
            print("Improvement detected, resetting early stopping patience.")
            self.early_stopping_tracker = 0
        else:
            self.early_stopping_tracker += 1
            print(f"No improvement. Incrementing Early Stopping tracker to {self.early_stopping_tracker}")

        return self.early_stopping_tracker > self.trainParams.early_stopping_patience

    def validation_improvement(self) -> bool:
        if self.trainParams.validation_metric_mode == "max":
            condition = (self.best_score < self.current_score)
        elif self.trainParams.validation_metric_mode == "min":
            condition = (self.best_score > self.current_score)
        else:
            raise ValueError("Invalid validation metric mode!")

        return condition

    def log(self, mode: str = "train", batch: int = None):
        if mode == "train":
            loss = self.train_loss / (batch + 1)
            metrics = self.train_metrics
            log_output = f'[Epoch {self.current_epoch}, Batch {batch + 1:5d}]: '
            step = self.current_epoch * (batch + 1)
        elif mode == "val":
            loss = self.val_loss
            metrics = self.val_metrics
            log_output = f'Validation Scores at epoch {self.current_epoch}: '
            step = self.current_epoch * len(self.train_loader)
        else:
            raise ValueError("Invalid mode!")

        # add loss first
        log_output += f"Loss: {loss:.4f}" + " | "
        self.logger.add_scalar(f"Loss/{mode}", loss, step)

        for metric in metrics.keys():
            if self.logger is not None:
                if metric.endswith("_class"):
                    # No verbosity for class metrics, only logging
                    # get scores
                    scores = metrics[metric].compute()
                    for i, label in enumerate(self.train_set.train_labels):
                        self.logger.add_scalar(f"{metric.capitalize()}/{label}/{mode}", scores[i], step)
                else:
                    # average metrics are computed, printed and logged
                    score = metrics[metric].compute()
                    log_output += f"{metric.capitalize()}: {score:.4f}" + " | "
                    self.logger.add_scalar(f"{metric.capitalize()}/{mode}", score, step)

        # remove final " | " before printing
        log_output = log_output[:-3]
        print(log_output)

        # delete all temporary variables
        try:
            del scores
        except UnboundLocalError as e:
            print("Error caught, error message:",e)
        try:
            del log_output
        except UnboundLocalError as e:
            print("Error caught, error message:",e)
        try:
            del step
        except UnboundLocalError as e:
            print("Error caught, error message:",e)

    def train_epoch(self):
        # training for a single epoch

        # switch model to training mode
        self.model.train()

        # reset metrics and loss at start of epoch
        self.train_loss = 0
        for metric in self.train_metrics.keys():
            self.train_metrics[metric].reset()

        #reset self.train_loader here for CL purposes
        self.train_loader = self.set_dataloaders(self.trainParams.batch_size, self.trainParams.num_workers,self.current_epoch+1,self.trainParams.n_epochs)[0]

        for batch, data in enumerate(self.train_loader):
            
            # extract data
            images = data["image"]
            labels = data["label"]

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

            # Update loss metrics after each batch
            self.train_loss += loss.item()
            for metric in self.train_metrics.keys():
                self.train_metrics[metric].update(pred, labels.int())

            # Logging
            if (batch + 1) % self.trainParams.update_steps == 0:
                self.log(mode="train", batch=batch)

        # clear memory
        del images, labels, loss, pred, batch, data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def validate(self):

        print(f'Validating at epoch {self.current_epoch}...')

        # Put model in eval mode
        self.model.eval()

        # # tensors to collect predictions and ground truths
        # predictions = torch.FloatTensor().to(self.device)
        # ground_truth = torch.FloatTensor().to(self.device)

        # reset validation loss and scores
        self.val_loss = 0
        for metric in self.val_metrics.keys():
            self.val_metrics[metric].reset()

        with torch.no_grad():
            for batch, data in enumerate(self.val_loader):
                # extract data
                images = data["image"]
                labels = data["label"]

                # move inputs to device
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward-pass
                output = self.model(images)
                loss = self.loss(output, labels)

                # update containers
                # ground_truth = torch.cat((ground_truth, labels), 0)
                # predictions = torch.cat((predictions, output), 0)

                # update validation loss and metric state after each batch
                self.val_loss += loss.item()
                for metric in self.val_metrics.keys():
                    self.val_metrics[metric].update(output, labels.int())

        # average loss
        self.val_loss /= len(self.val_loader)

        # Logging
        self.log(mode="val")

        # Clear memory
        del images, labels, output, loss, batch, data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def train(self):
        # set seed
        print(f"Device set to {self.device}.")

        torch.manual_seed(self.seed)
        print(f"Seed set to {self.seed}.")

        print(f'Starting training at path {self.work_dir}.')

        # set model to device
        self.model.to(self.device)

        for epoch in range(self.trainParams.n_epochs):
            self.current_epoch += 1

            # Training
            print(f'Training at epoch {self.current_epoch}/{self.trainParams.n_epochs}...')
            self.train_epoch()

            # Validation
            self.validate()

            if self.trainParams.validation_metric != "loss":
                self.current_score = self.val_metrics[self.trainParams.validation_metric].compute()
            else:
                self.current_score = self.val_loss

            # update learning rate
            if self.trainParams.lr_policy == "plateau":
                self.lr_scheduler.step(self.current_score)
            elif self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # save model on epoch
            if (self.current_epoch % self.trainParams.save_epoch_freq) == 0:
                self.save_model(save_best=False)

            improved = self.validation_improvement()
            # save best model
            if improved:
                print(f'Model improved at epoch {self.current_epoch}, saving model.')
                self.best_score = self.current_score
                self.save_model(save_best=True)
            # early stopping
            if self.trainParams.early_stopping:
                if self.check_early_stopping(improved):
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

    def save_params(self):
        # convert params to dicts
        params = {'trainParams': self.trainParams.to_dict(),
                  'modelParams': self.modelParams.to_dict(),
                  'dataTrainParams': self.dataTrainParams.to_dict(),
                  'dataValParams': self.dataValParams.to_dict(),
                  'augmentParams': self.augmentParams.to_dict()}

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
                    "validation_metric": self.trainParams.validation_metric,
                    "score": self.current_score,
                    "best_score": self.best_score}, save_path)

    def load_model(self):
        # find the latest checkpoint in the work_dir that does not contain 'best' in the name
        checkpoints = [f for f in os.listdir(self.trainParams.work_dir) if f.endswith('.pth') and 'best' not in f]
        if not checkpoints:
            raise FileNotFoundError("No checkpoint found in %s" % self.trainParams.work_dir)

        checkpoints = natsorted(checkpoints)
        load_path = os.path.join(self.trainParams.work_dir, checkpoints[-1])

        loaded_dict = torch.load(load_path)

        self.modelParams = loaded_dict["modelParams"]
        self.model = load_model(self.modelParams)
        self.model.load_state_dict(loaded_dict["model"])
        self.optimizer.load_state_dict(loaded_dict["optimizer"])
        self.current_epoch = loaded_dict["epoch"]

        # update best score and current score depending on new settings
        validation_metric = loaded_dict["validation_metric"]
        if self.trainParams.validation_metric != validation_metric:
            print(f"Detected validation metric {validation_metric} in checkpoint, which is different from "
                  f"current validation metric {self.trainParams.validation_metric}. Resetting scores.")
            if self.trainParams.validation_metric_mode == "max":
                self.best_score = 0
                self.current_score = 0
            elif self.trainParams.validation_metric_mode == "min":
                self.best_score = np.infty
                self.current_score = np.infty
        else:
            self.best_score = loaded_dict["best_score"]
            self.current_score = loaded_dict["score"]
