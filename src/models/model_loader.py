from pretrained_model import PretrainedModel
# from custom_model import CustomModel

from params import PretrainedModelParams, CustomModelParams


def load_model(modelParams):
    if modelParams.model_type == "pretrained":
        return PretrainedModel(modelParams)
    # elif modelParams.model_type == "custom":
    #    return CustomModel(modelParams)
    else:
        raise ValueError(f"Unknown model type: {modelParams.model_type}")
