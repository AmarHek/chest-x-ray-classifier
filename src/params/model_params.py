from dataclasses import dataclass, field
from typing import List

from params.base_params import BaseParams


@dataclass
class ModelParams(BaseParams):
    name = "Model Parameters"

    model_type = "pretrained"  # [pretrained | custom]


@dataclass
class PretrainedModelParams(ModelParams):
    name = "Pretrained Model Parameters"

    model_name: str = "densenet121"

    # freeze layers
    freeze: bool = False
    # 0 = freeze all layers, 1 = freeze all but last layer, etc., the way of freezing depends on the model
    freeze_depth: int = 0

    # classifier
    linear_weights: List[int] = field(default_factory=lambda: [512])  # how many neurons in each linear layer
    activation_function: str = "relu"  # [relu | leaky_relu | sigmoid | tanh]
    dropout: List[float] = field(default_factory=lambda: [0.5])  # dropout rate for each linear layer
    # length must be the same as linear_weights, set to 0 to disable dropout for a layer
    classifier_function: str = "sigmoid"  # [sigmoid | softmax]
