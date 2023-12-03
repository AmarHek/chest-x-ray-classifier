from dataclasses import dataclass, field
from typing import List

from params.base_params import BaseParams


@dataclass
class ModelParams(BaseParams):
    name = "Model Parameters"
    model_type = "pretrained"  # [pretrained | custom]


@dataclass
class PretrainedModelParams(ModelParams):
    backbone: str = "densenet121"
    weights: str = "DEFAULT"  # [DEFAULT | IMAGENET1K_V1 | IMAGENET1K_V2]

    # -1 = freeze no layers, 0 = freeze all layers,
    # 1 = freeze all but last layer, etc.
    freeze_depth: int = -1

    # dropout
    dropout: float = 0.0  # dropout is applied after avg pool

    # classifier
    head: str = "linear"  # [linear | csra | none]
    classifier_function: str = "sigmoid"  # [sigmoid | softmax | logsoftmax | logsigmoid]
    num_classes: int = 1

    # csra params
    num_heads: int = 1
    lam: float = 0.1

    # more classifier params can be added here


@dataclass
class CustomModelParams(ModelParams):
    model_name: str = "custom"
    # Custom Model Parameters can be added here
