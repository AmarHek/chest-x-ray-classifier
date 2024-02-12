import torchvision
from dataclasses import dataclass

from params.base_params import BaseParams


@dataclass
class ModelParams(BaseParams):
    name = "Model Parameters"
    model_type: str = "base"  # [pretrained | custom]


@dataclass
class PretrainedModelParams(ModelParams):
    name = "Pretrained Model Parameters"
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


@dataclass
class HuggingfaceModelParams(ModelParams):
    name = "Huggingface Model Parameters"
    backbone: str = "facebook/dinov2-base"

    # classifier
    head: str = "linear"  # [linear | csra | none]
    classifier_function: str = "sigmoid"  # [sigmoid | softmax | logsoftmax | logsigmoid]
    num_classes: int = 1

    # csra params
    num_heads: int = 1
    lam: float = 0.1


@dataclass
class CustomModelParams(ModelParams):
    name = "Custom Model Parameters"
    model_name: str = "custom"
    # Custom Model Parameters can be added here


model_param_selector = {
    "pretrained": PretrainedModelParams,
    "huggingface": HuggingfaceModelParams,
    "custom": CustomModelParams
}
