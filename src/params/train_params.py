from dataclasses import dataclass, field
from typing import Tuple, List

from params.base_params import BaseParams


@dataclass
class TrainParams(BaseParams):
    name = "Training Parameters"

    exp_name: str = "Experiment"
    work_dir: str = "./work_dirs/"

    seed: int = 42069

    # Logging
    # TODO implement wandb
    logger: str = "tensorboard"  # [tensorboard | wandb | None]
    update_steps: int = 100  # how often to update the logger
    save_epoch_freq: int = 1
    max_keep_ckpts: int = 3  # how many checkpoints to save (beside best)

    # general parameters
    n_epochs: int = 100
    continue_train: bool = False
    batch_size: int = 32

    # hardware
    device: str = "cuda"
    num_workers: int = 4

    # loss and optimizer
    loss: str = 'bce'
    optimizer: str = 'adam'
    learning_rate: float = 0.0002
    # which metrics to track during training
    metrics: List[str] = field(default_factory=lambda: ["loss", "auc", "prec", "rec", "f1"])

    # validation parameters
    validation_metric: str = "loss"  # which metric to use for saving best model
    validation_metric_mode: str = "min"  # [min | max]
    validation_epoch_freq: int = 1
    validation_batch_size: int = 32

    # early stopping
    early_stopping: bool = False
    early_stopping_patience: int = 10

    # learning rate schedulers
    lr_policy: str = "plateau"  # [linear | step | plateau | cosine | cyclic | exponential]
    lr_decay_iters: int = 50
    n_epochs_decay: int = 100  # linear learning rate decay
    plateau_patience: int = 5
    exponential_gamma: float = 0.01
    cyclic_lr: Tuple[float, float] = (0.001, 0.01)
