import torch
from transformers import AutoModelForImageClassification
from torch import nn

from models import classifier_head, classifier_function
from src.params.model_params import HuggingfaceModelParams


class HuggingfaceModel(nn.Module):

    def __init__(self, model_params: HuggingfaceModelParams):

        super(HuggingfaceModel, self).__init__()
        self.model_params = model_params

        self.base_model = AutoModelForImageClassification.from_pretrained(self.model_params.model_name)
        num_features = self.base_model.classifier.in_features

        head = classifier_head(in_features=num_features, **self.model_params.to_dict())
        cls_function = classifier_function(self.model_params.classifier_function)
        self.base_model.classifier = nn.Sequential(head, cls_function)

    def forward(self, x):
        x = self.base_model(x)
        return x


if __name__ == "__main__":
    dummy_input = torch.zeros(4, 3, 224, 224)

    model_params = HuggingfaceModelParams()
    model_params.model_name = "microsoft/swinv2-base-patch4-window16-256"
    model_params.head = "csra"
    model_params.num_heads = 4
    model_params.lam = 0.1
    model_params.num_classes = 14

    model = HuggingfaceModel(model_params)
    print(model)

    print(model(dummy_input))