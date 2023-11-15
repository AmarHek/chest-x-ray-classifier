import os.path
import json
from typing import Union

import torch.cuda
from torch.utils.data import Dataset, DataLoader

from src.functions import metrics_selector as metrics_dict, metric_is_valid
from src.util.util import get_file_contents_as_list, serialize_numpy_array


class Tester:
    def __init__(self,
                 test_set: Dataset,
                 classes: list[str],
                 model_paths: Union[str, list[str]] = None,
                 models_file: str = None,
                 metrics=None,
                 threshold: float = 0.5,
                 batch_size: int = 10,
                 ensemble: bool = False):

        if model_paths is None and models_file is None:
            raise ValueError("model_path and models_file cannot both be None, "
                             "specify at least one!")

        self.test_set = test_set
        self.test_loader = DataLoader(self.test_set, batch_size=batch_size, num_workers=1,
                                      shuffle=False, drop_last=False)
        # TODO: implement test time augmentation

        self.classes = classes
        self.threshold = threshold
        print("Threshold set to 0.5.")
        self.ensemble = ensemble
        if ensemble:
            print("Ensemble mode enabled")

        self.model_paths: list[str] = []
        self.model_names: list[str] = []
        self.outputs: dict = {}
        self.metrics_result: dict = {}

        if metrics is None:
            self.compute_metrics = False
            self.metrics = []
        else:
            for metric in metrics:
                if not metric_is_valid(metric):
                    raise ValueError("%s is an invalid metric!" % metric)
            self.metrics = metrics
            self.compute_metrics = True

        if model_paths is not None:
            if type(model_paths) == str:
                self.add_model(model_paths)
            else:
                for path in model_paths:
                    self.add_model(path)
        if models_file is not None:
            self.add_models_from_csv(models_file)

        self.set_device()

    def set_device(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device set to {self.device}.")

    @staticmethod
    def load_model(model_path: str):
        return torch.load(model_path)

    def reset(self):
        self.model_paths = []
        self.model_names = []
        self.outputs = {}
        self.metrics_result = {}

    def add_model(self, model_path: str):
        assert model_path.endswith('.pt'), "Only .pt allowed for testing!"
        assert os.path.basename(model_path) not in self.model_names, \
            "%s already exists in list of models!" % os.path.basename(model_path)
        self.model_paths.append(model_path)
        self.model_names.append(os.path.basename(model_path))

    def add_models_from_list(self, model_paths: list[str]):
        assert len(model_paths) > 0, "Specify at least one model!"
        for model_path in model_paths:
            self.add_model(model_path)

    def add_models_from_csv(self, models_file_path):
        assert os.path.isfile(models_file_path), "Invalid path given!"
        model_paths = get_file_contents_as_list(models_file_path)
        for model_path in model_paths:
            self.add_model(model_path)

    def test_single_model(self, idx):
        # get paths, names and load the model into memory
        model_path, model_name = self.model_paths[idx], self.model_names[idx]
        model = Tester.load_model(model_path)
        model.to(self.device)
        print("Now testing model %s" % model_name)
        # set model to evaluate mode
        model.eval()

        # tensors to collect predictions and ground truths
        predictions = torch.FloatTensor().to('cpu')
        ground_truth = torch.FloatTensor().to('cpu')

        with torch.no_grad():
            # first iterate through dataloader and collect all outputs
            for (images, labels) in self.test_loader:
                # move image to device
                images = images.to(self.device)
                labels = labels.to(self.device)

                output = model(images)

                # update containers
                ground_truth = torch.cat((ground_truth, labels.to('cpu')), 0)
                predictions = torch.cat((predictions, output.to('cpu')), 0)

                del images, labels
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # save raw outputs to dict
        self.outputs[model_name] = serialize_numpy_array(predictions.numpy())

        # compute metrics
        self.metrics_result[model_name] = self.compute_metrics(ground_truth, predictions)

        # Clear memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test(self):
        if not self.ensemble:
            for idx, model in enumerate(self.model_paths):
                self.test_single_model(idx)
        # TODO: implement ensemble computation

    def compute_metrics(self, y_gt, y_pred):
        result = {}
        for metric in self.metrics:
            metric_function = metrics_dict[metric]
            if metric == "auroc":
                results_per_class = metric_function(y_gt, y_pred)
                result["avg_" + metric] = metric_function(y_gt, y_pred, average="macro")
            else:
                results_per_class = metric_function(y_gt, y_pred, threshold=self.threshold)
                result["avg_" + metric] = metric_function(y_gt, y_pred, threshold=self.threshold, average="macro")

            class_results = {}
            for (class_name, res) in zip(self.classes, results_per_class):
                class_results[class_name] = res
            result[metric + "_per_class"] = class_results

        return result

    def save_metrics(self, save_path):
        with open(save_path, "w") as sp:
            json.dump(self.metrics_result, sp)

    def save_raw_results(self, save_path):
        with open(save_path, "w") as sp:
            json.dump(self.outputs, sp)
