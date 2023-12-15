from typing import List
from dataclasses import dataclass, field

from params import BaseParams


@dataclass
class TestParams(BaseParams):
    name = "Test Parameters"

    work_dir: str = "./work_dirs/"  # where are the models saved
    # Two options: either search recursively for models in work_dir or only search in work_dir + experiment
    recursive: bool = False  # whether to recursively search for models in work_dir
    exp_names: str | List[str] = "Experiment"  # name of the experiment dir(s)

    compute_metrics: bool = True
    metrics: List[str] = field(default_factory=lambda: ["auc", "prec", "rec", "f1"])
    threshold: float = 0.5

    batch_size: int = 32
    test_time_augmentation: bool = False  # whether to use test time augmentation
