import torch
from torch import nn
from torch.nn import Sequential

from params.model_params import ModelParams
from models.model_components import get_backbone


class PretrainedModel(nn.Module):

    def __init__(self, model_params: ModelParams):

        super(PretrainedModel, self).__init__()
        self.model_params = model_params

        self.backbone = get_backbone(self.model_params.backbone, self.model_params.weights)

        self.classifier = self.get_classifier()

    @staticmethod
    def get_num_features(model, input_tensor):
        # Extract the feature part of the model up to the final pooling layer
        features = Sequential(*list(model.children())[:-1])

        # Put the feature part and a dummy input through the model to get the output size
        with torch.no_grad():
            output = features(input_tensor)

        # Return the number of output features
        return output.view(output.size(0), -1).shape[1]


if __name__ == "__main__":
    import torchvision.models as tvmodels

    model = tvmodels.resnet50()
    backbone = Sequential(*list(model.children())[:-1])
    print(backbone)
