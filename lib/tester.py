import os.path

import torch.nn as nn
import torch.cuda
from torch.utils.data import DataLoader
import libauc
from tqdm import tqdm

from lib.dataset import ChestXray
from lib.metrics import multi_label_auroc


class Tester:

    def __init__(self,
                 model_path: str or list[str],
                 test_set: ChestXray,
                 metrics: list[str],
                 threshold: float = 0.5):

        self.model: list[nn.Module] = []

    @staticmethod
    def load_model(model_path: str) -> nn.Module:
        return torch.load(model_path)

    def add_model(self, model_path):
        assert os.path.isfile(model_path), "Invalid model_path given!"
        self.model.append(Tester.load_model(model_path))

