import os.path
from typing import List

import pandas as pd
import torch.cuda
from torch.utils.data import DataLoader
import torchvision.transforms as tfs

from components import load_metrics
from datasets import load_dataset
from params import TestParams, DatasetParams, AugmentationParams
from models import load_model


class Tester:
    def __init__(self,
                 testParams: TestParams,
                 datasetParams: DatasetParams,
                 augmentParams: AugmentationParams = None):

        # get proper device
        if torch.cuda.is_available() and testParams.device == "cuda":
            self.device = "cuda"
        else:
            print("CUDA not available, using CPU.")
            self.device = "cpu"
        print(f"Device set to {self.device}.")

        # some checks for overlapping parameters
        if testParams.test_time_augmentation and augmentParams is None:
            raise ValueError("AugmentParams must be provided if test_time_augmentation is True.")
        if testParams.automatic and testParams.model_paths is not None:
            print("Warning: model_paths will be ignored if automatic is True.")

        # get work_dir
        if os.path.exists(testParams.work_dir):
            self.work_dir = testParams.work_dir
        else:
            raise ValueError(f"Work directory {testParams.work_dir} does not exist.")

        # get name of output dir
        if testParams.output_dir == "":
            self.output_dir = datasetParams.dataset
            print(f"Output directory name set to {self.output_dir}.")
        else:
            self.output_dir = testParams.output_dir
            print(f"Output directory name set to {self.output_dir}.")

        # create dataset and check for test time augmentation
        if testParams.test_time_augmentation:
            self.tta = True
            self.augmentations = self.test_time_augmentations(augmentParams)
            self.num_augmentations = testParams.num_augmentations
        else:
            self.tta = False
        self.test_set = load_dataset(datasetParams, None)
        self.test_loader = DataLoader(self.test_set, batch_size=testParams.batch_size,
                                      num_workers=1, shuffle=False, drop_last=False)
        self.labels = self.test_set.train_labels

        # get model paths
        print("Getting model paths...")
        self.overwrite = testParams.overwrite
        self.automatic = testParams.automatic
        self.write_filenames = testParams.write_filenames
        self.model_paths = self.get_model_paths(testParams.model_paths)
        print(f"Found {len(self.model_paths)} models.")

        # create metrics
        print("Creating metrics...")
        self.threshold = testParams.threshold
        if testParams.metrics is not None:
            self.metrics = load_metrics(testParams.metrics,
                                        num_classes=len(self.labels),
                                        threshold=self.threshold,
                                        task='multilabel',
                                        device=self.device)
            self.compute_metrics = True
            self.test_scores: dict = {}
        else:
            self.compute_metrics = False
            print("Warning: No metrics will be computed.")
        print(f"Metrics set to {testParams.metrics}.")

        # initialize trackers
        self.predictions = torch.FloatTensor().to(self.device)
        self.ground_truth = torch.FloatTensor().to(self.device)
        self.filenames = []

    def _outputs_exist(self, model_path: str):
        if os.path.exists(os.path.join(model_path, self.output_dir)):
            # check for outputs.csv and metrics.json
            if os.path.exists(os.path.join(model_path, self.output_dir, "outputs.csv")) and \
                    os.path.exists(os.path.join(model_path, self.output_dir, "metrics.json")):
                return True
            else:
                return False
        else:
            return False

    @staticmethod
    def _best_model_exists(model_path: str):
        # search for a file ending on 'best.pth' within the model_path
        for file in os.listdir(model_path):
            if file.endswith(".pth") and "best" in file:
                return True
        return False

    def get_model_paths(self, model_paths: List[str] = None):
        if self.automatic:
            return self.get_model_paths_automatic()
        else:
            if model_paths is None:
                raise ValueError("If automatic is False, model_paths must be provided.")
            else:
                return self.get_model_paths_from_list(model_paths)

    def get_model_paths_automatic(self):
        print(f"Inferring model paths from work_dir {self.work_dir} automatically.")
        model_paths = []
        for root, dirs, files in os.walk(self.work_dir):
            if self._best_model_exists(root):
                if self._outputs_exist(root):
                    print(f"Found existing model in {root}.")
                    if self.overwrite:
                        print(f"Found existing outputs in {root}, but overwrite is enabled, adding to model_paths.")
                        model_paths.append(root)
                    else:
                        print(f"Found existing outputs in {root}, skipping.")
                else:
                    print(f"Found existing model in {root}, but no outputs, adding to model_paths.")
                    model_paths.append(root)

        return model_paths

    def get_model_paths_from_list(self, model_paths: List[str]):
        model_paths_filtered = []
        for model_path in model_paths:
            if os.path.isdir(os.path.join(self.work_dir, model_path)):
                if self._best_model_exists(os.path.join(self.work_dir, model_path)):
                    print(f"Found existing model in {model_path}.")
                    if self._outputs_exist(os.path.join(self.work_dir, model_path)):
                        if self.overwrite:
                            print(f"Found existing outputs in {model_path}, but overwrite is enabled, "
                                  f"adding to model_paths.")
                            model_paths_filtered.append(os.path.join(self.work_dir, model_path))
                        else:
                            print(f"Found existing outputs in {model_path}, skipping.")
                    else:
                        print(f"Found existing model in {model_path}, but no outputs, adding to model_paths.")
                        model_paths_filtered.append(os.path.join(self.work_dir, model_path))
                else:
                    print(f"No model found in {model_path}, skipping.")
            else:
                print(f"Model path {model_path} is not a directory, skipping.")

        return model_paths_filtered

    def load_model_from_path(self, model_path: str):
        # load model
        best_checkpoint = None
        for file in os.listdir(model_path):
            if file.endswith(".pth") and "best" in file:
                best_checkpoint = os.path.join(model_path, file)
        if best_checkpoint is None:
            raise ValueError(f"No best checkpoint found in {model_path}.")

        model_data = torch.load(best_checkpoint)
        model = load_model(model_data["modelParams"])
        model.load_state_dict(model_data["model"])
        model.to(self.device)
        model.eval()

        return model

    # For test time augmentation
    def test_time_augmentations(self, augmentParams: AugmentationParams):
        transforms = []

        # denormalize images first
        if self.test_set.__mean__ is not None and self.test_set.__std__ is not None:
            transforms.append(
                tfs.Normalize(mean=[-m / s for m, s in zip(self.test_set.__mean__, self.test_set.__std__)],
                              std=[1 / s for s in self.test_set.__std__]))
        # descale from 0-1 to 0-255
        transforms.append(tfs.Lambda(lambda x: x * 255))
        # convert to PIL image
        transforms.append(tfs.ToPILImage())

        # append specified augmentations
        if augmentParams.randomAffine:
            transforms.append(tfs.RandomAffine(degrees=augmentParams.degrees,
                                               scale=augmentParams.scale,
                                               translate=augmentParams.translate,
                                               fill=augmentParams.fill))
        if augmentParams.randomResizedCrop:
            transforms.append(tfs.RandomResizedCrop(size=augmentParams.crop_scale, ratio=(1, 1)))
        if augmentParams.horizontalFlip:
            transforms.append(tfs.RandomHorizontalFlip(0.5))
        if augmentParams.verticalFlip:
            transforms.append(tfs.RandomVerticalFlip(0.5))

        # append standard test transforms
        transforms.append(tfs.Resize(self.test_set.image_size))
        transforms.append(tfs.ToTensor())
        if self.test_set.__mean__ is not None and self.test_set.__std__ is not None:
            transforms.append(tfs.Normalize(mean=self.test_set.__mean__, std=self.test_set.__std__))

        return tfs.Compose(transforms)

    def apply_batch_augmentation(self, batch):
        augmented_batch = torch.stack([self.augmentations(img) for img in batch])
        return augmented_batch

    def _create_output_dir(self, model_path: str):
        output_dir = os.path.join(model_path, self.output_dir)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        return output_dir

    def write_output(self, model_dir: str):
        # save outputs as csv with labels as columns
        file = os.path.join(model_dir, self.output_dir, "outputs.csv")
        predictions_np = self.predictions.cpu().numpy()
        predictions_df = pd.DataFrame(predictions_np, columns=self.labels)
        if self.write_filenames:
            filenames = pd.Series(self.filenames)
            predictions_df.insert(0, "filename", filenames)
        predictions_df.to_csv(file, index=False)

    def update_metrics(self):
        for metric in self.metrics.keys():
            if metric.endswith('_class'):
                self.test_scores[metric] = self.metrics[metric](self.predictions, self.ground_truth.int()).cpu().numpy()
            else:
                self.test_scores[metric] = self.metrics[metric](self.predictions, self.ground_truth.int()).item()

    def write_metrics(self, model_dir: str):
        # save metrics as csv
        file = os.path.join(model_dir, self.output_dir, "metrics.csv")

        # Separate average and class-specific metrics
        average_metrics = {}
        class_metrics = {}

        for metric, value in self.test_scores.items():
            if metric.endswith('_class'):
                class_metrics[metric.replace('_class', '')] = value
            else:
                average_metrics[metric] = value

        # Create DataFrames for average and class-specific metrics
        # average is straightforward
        df_average = pd.DataFrame(list(average_metrics.items()), columns=['Metric', 'Average'])
        # for class, we first create an empty DataFrame
        df_class = pd.DataFrame(columns=['Metric'] + self.labels)
        # populate the DataFrame
        for metric, values in class_metrics.items():
            temp_df = pd.DataFrame({'Metric': metric, **dict(zip(self.labels, values))}, index=[0])
            df_class = pd.concat([df_class, temp_df], ignore_index=True)

        # Concatenate DataFrames along columns
        result_df = pd.merge(left=df_average, right=df_class, on='Metric', how='outer')

        # Replace NaN with -1
        result_df = result_df.fillna(-1)

        # Save to csv
        result_df.to_csv(file, index=False)

    def test_model(self, model):
        # reset containers
        self.predictions = torch.FloatTensor().to(self.device)
        self.ground_truth = torch.IntTensor().to(self.device)
        self.filenames = []

        with torch.no_grad():
            for data in self.test_loader:
                # extract data
                images = data['image']
                labels = data['label']
                fns = data['image_path']

                # move inputs to device
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Get regular output
                output = model(images)

                # Get augmented output if TTA is enabled
                if self.tta:
                    for _ in range(self.num_augmentations - 1):
                        augmented_images = self.apply_batch_augmentation(images)
                        augmented_output = model(augmented_images)
                        output += augmented_output
                    output /= (self.num_augmentations + 1)

                # update containers
                self.ground_truth = torch.cat((self.ground_truth, labels), 0)
                self.predictions = torch.cat((self.predictions, output), 0)
                self.filenames.extend(fns)

    def test(self):
        print(f"Testing {len(self.model_paths)} models...")

        for model_path in self.model_paths:
            print(f"Testing model {model_path}...")
            print(f"Setting up output directory...")
            self._create_output_dir(model_path)
            print(f"Loading model...")
            model = self.load_model_from_path(model_path)
            print(f"Running inference...")
            self.test_model(model)
            print(f"Saving outputs...")
            self.write_output(model_path)
            if self.compute_metrics:
                self.update_metrics()
                self.write_metrics(model_path)
                # reset torchmetrics metrics
                for metric in self.metrics.keys():
                    self.metrics[metric].reset()
            print("Done.")

        print("Testing done.")
