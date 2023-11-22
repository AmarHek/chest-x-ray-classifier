from dataclasses import dataclass
from typing import Tuple

from params.base_params import BaseParams


@dataclass
class TrainParams(BaseParams):
    name = "Training Parameters"

    exp_name: str = "Experiment"
    work_dir: str = "./work_dirs/"

    seed: int = 42069

    # Logging
    write_logs: bool = True
    logger: str = "tensorboard"  # [tensorboard | wandb]
    update_steps: int = 1000

    # epoch parameters
    n_epochs: int = 100
    save_epoch_freq: int = 1
    max_keep_ckpts: int = 3  # how many checkpoints to save (beside best)
    validate: bool = False
    continue_train: bool = False

    # dataset parameters
    batch_size: int = 32
    num_workers: int = 4

    # loss and optimizer parameters
    loss: str = 'bce'
    optimizer: str = 'adam'
    learning_rate: float = 0.0002
    # TODO Add more optimizer parameters

    # validation parameters
    validation_metrics: str = "loss"
    early_stopping: bool = False
    early_stopping_patience: int = 10

    # learning rate schedulers
    lr_policy: str = "plateau"  # [linear | step | plateau | cosine]
    lr_decay_iters: int = 50
    n_epochs_decay: int = 100  # linear learning rate decay
    plateau_patience: int = 5
    exponential_gamma: float = 0.01
    cyclic_lr_min: Tuple[float, float] = (0.001, 0.01)
