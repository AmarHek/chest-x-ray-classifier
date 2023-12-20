import os.path
import json
from typing import Union

import torch.cuda
from torch.utils.data import Dataset, DataLoader

from components import load_metrics
from datasets import load_dataset
from src.utils.util import get_file_contents_as_list, serialize_numpy_array


class Tester:
    def __init__(self,
                 testParams: TestParams,
                 datasetParams: DatasetParams,
                 augmentParams: AugmentParams = None):

        self.model_paths = []
        self.test_set = load_dataset(datasetParams, augmentParams)
        self.test_loader = DataLoader(self.test_set, batch_size=testParams.batch_size,
                                      num_workers=1, shuffle=False, drop_last=False)

        self.labels = self.test_set.train_labels
        self.threshold = testParams.threshold
        self.compute_metrics = testParams.compute_metrics

        self.model_paths: list[str] = []
        self.model_names: list[str] = []
        self.outputs: dict = {}
        self.metrics_result: dict = {}

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device set to {self.device}.")


