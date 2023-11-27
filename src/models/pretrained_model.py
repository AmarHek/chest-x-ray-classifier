from torch import nn
import torchvision.models as tvmodels
from params.model_params import ModelParams


class PretrainedModel(nn.Module):

    def __init__(self, model_params: ModelParams):

        super(PretrainedModel, self).__init__()
        self.model_params = model_params

        self.backbone = self.get_pretrained_model()

        self.classifier = self.get_classifier()

    @staticmethod
    def remove_classifier(backbone):
        # Check if the model has a classifier (fc) attribute
        if hasattr(backbone, 'fc'):
            del backbone.fc
        elif hasattr(backbone, 'classifier'):
            del backbone.classifier
        else:
            raise ValueError("Unsupported model architecture. Please modify the code accordingly.")

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
    # model = tvmodels.densenet121(weights="DEFAULT")
    model = tvmodels.efficientnet_b2(weights="DEFAULT")
    # model2 = tvmodels.vit_l_32(weights="DEFAULT")
    print(model)

