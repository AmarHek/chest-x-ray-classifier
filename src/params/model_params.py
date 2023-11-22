from dataclasses import dataclass

from params.base_params import BaseParams


@dataclass
class ModelParams(BaseParams):
    name = "Model Parameters"

    model_type = "cnn"  # [cnn | transformer]

