import os.path

import torch.nn as nn
import torch.cuda

from torch.utils.data import Dataset, DataLoader
from functions import metrics as metrics_dict, metric_is_valid
from util.util import get_file_contents_as_list


class Tester:
    def __init__(self,
                 test_set: Dataset,
                 classes: list[str],
                 model_path: str = None,
                 models_file: str = None,
                 metrics="auroc",
                 threshold: float = 0.5):

        if model_path is None and models_file is None:
            raise ValueError("model_path and models_file cannot both be None, "
                             "specify at least one!")

        self.test_set = test_set
        self.test_loader = DataLoader(self.test_set, batch_size=1, num_workers=1,
                                      shuffle=False, drop_last=False)
        self.classes = classes
        self.threshold = threshold
        print("Threshold set to 0.5.")

        self.models: list[nn.Module] = []
        self.model_names: list[str] = []
        self.metrics: list = []
        self.metric_names = metrics
        self.outputs: dict = {}
        self.computed_metrics: dict = {}

        if model_path is not None:
            self.load_model(model_path)

        if models_file is not None:
            self.load_models_from_csv(models_file)

        if type(metrics) == str:
            if metric_is_valid(metrics):
                self.metrics.append(metrics_dict[metrics])
        else:
            for metric in metrics:
                if metric_is_valid(metric):
                    self.metrics.append(metrics_dict[metric])

        self.init_output_dicts()

    def set_device(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device set to {self.device}.")

    def load_model(self, model_path: str):
        assert model_path.endswith('.pt'), "Only .pt allowed for testing!"
        assert os.path.basename(model_path) not in self.model_names, \
            "%s is already loaded!" % os.path.basename(model_path)
        self.models.append(torch.load(model_path))
        self.model_names.append(os.path.basename(model_path))

    def load_models_from_csv(self, models_file_path):
        assert os.path.isfile(models_file_path), "Invalid model_path given!"
        model_paths = get_file_contents_as_list(models_file_path)
        for model_path in model_paths:
            self.load_model(model_path)

    def init_output_dicts(self):
        for model_name in self.model_names:
            self.outputs[model_name] = []
            metrics_output = {}
            for metric in self.metric_names:
                metrics_output[metric + "_per_class"] = []
                metrics_output["avg_" + metric] = 0
            self.computed_metrics[model_name] = metrics_output

    def test_single_model(self, idx):
        model, model_name = self.models[idx], self.model_names[idx]
        print("Now testing model %s" % model_name)
        # set model to evaluate mode
        model.eval()

        predictions = []
        ground_truth = []

        # first iterate through dataloader and collect all outputs
        for (image, labels) in self.test_loader:
            # move image to device
            image = image.to(self.device)

            pred = model(image)

            # append prediction and ground_truth
            predictions.append(pred)
            ground_truth.append(labels)

        self.outputs[model_name] = predictions

    def test(self):
        for idx, model in enumerate(self.models):
            self.test_single_model(idx)

    def save_metrics(self, save_path):
        pass

    def save_raw_results(self, save_path):
        pass
