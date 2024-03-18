from .huggingface_model import HuggingfaceModel
from .pretrained_model import PretrainedModel
# from custom_model import CustomModel


def load_model(modelParams):
    if modelParams.model_type == "pretrained":
        return PretrainedModel(modelParams)
    elif modelParams.model_type == "huggingface":
        return HuggingfaceModel(modelParams)
    # elif modelParams.model_type == "custom":
    #    return CustomModel(modelParams)
    else:
        raise ValueError(f"Unknown model type: {modelParams.model_type}")
