from typing import List
from dataclasses import dataclass, field

from params import BaseParams


@dataclass
class TestParams(BaseParams):
    name = "Test Parameters"

    # Two options: either search automatically for models in work_dir or
    # only search in work_dir + model_paths
    automatic: bool = False  # whether to auto search for models (and results) in work_dir
    work_dir: str = "./work_dirs/"  # where are the models saved
    model_paths: str | List[str] = field(default_factory=lambda: [])  # name(s) of the model subdirectories
    output_dir: str = ""  # name of the output dir (if empty, then name is derived from dataset name)
    overwrite: bool = False  # whether to overwrite existing results
    write_filenames: bool = True  # whether to write filenames to output.csv

    device: str = "cuda"
    metrics: List[str] = field(default_factory=lambda: [])  # which metrics to compute
    threshold: float = 0.5

    batch_size: int = 32
    test_time_augmentation: bool = False  # whether to use test time augmentation
    num_augmentations: int = 5  # how many augmented images to use
