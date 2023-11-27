

def load_model(modelParams):

    if modelParams.model_type == "pretrained":
        model = PretrainedModel(modelParams)
    else:
