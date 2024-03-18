import torch
from transformers import AutoModel
from torch import nn

from models import classifier_head, classifier_function
from src.params.model_params import HuggingfaceModelParams


class HuggingfaceModel(nn.Module):

    def __init__(self, model_params: HuggingfaceModelParams):

        super(HuggingfaceModel, self).__init__()
        self.model_params = model_params

        self.backbone = AutoModel.from_pretrained(self.model_params.backbone)
        num_features = self.get_num_features()

        # for now, only allow linear head
        # TODO implementation for other heads
        self.model_params.head = "linear"
        head = classifier_head(in_features=num_features, **self.model_params.to_dict(), linear_input=True)
        cls_function = classifier_function(self.model_params.classifier_function)
        self.classifier = nn.Sequential(head, cls_function)

    def forward(self, x):
        outputs = self.backbone(x)
        classifier_input = self.get_classifier_input(outputs)
        logits = self.classifier(classifier_input)

        return logits

    def get_num_features(self) -> int:
        dummy_input = torch.zeros(1, 3, 224, 224)
        # Run a forward pass of the dummy input tensor through the feature extractor
        with torch.no_grad():
            output = self.backbone(dummy_input)
        adjusted_output = self.get_classifier_input(output)

        # Return the channel dimension of the output
        return adjusted_output.shape[1]

    def get_classifier_input(self, transformer_output):
        # depending on the backbone transformer, the output must be handled differently
        if any(backbone in self.model_params.backbone for backbone in ["swin", "convnext"]):
            classifier_input = transformer_output[1]
        elif "dino" in self.model_params.backbone:
            sequence_output = transformer_output[0]  # batch_size, sequence_length, hidden_size
            cls_token = sequence_output[:, 0]
            patch_tokens = sequence_output[:, 1:]
            classifier_input = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1)
        elif "efficientformer" in self.model_params.backbone:
            sequence_output = transformer_output[0]
            classifier_input = sequence_output.mean(-2)
        elif any(backbone in self.model_params.backbone for backbone in ["vit", "pvt"]):
            sequence_output = transformer_output[0]
            classifier_input = sequence_output[:, 0, :]
        else:
            raise ValueError(f"Backbone {self.model_params.backbone} not supported")

        return classifier_input


if __name__ == "__main__":
    dummy_input = torch.zeros(4, 3, 224, 224)

    model_params = HuggingfaceModelParams()
    model_params.backbone = "microsoft/swinv2-base-patch4-window16-256"
    model_params.head = "csra"
    model_params.num_classes = 14

    model = HuggingfaceModel(model_params)
    print(model)

    print(model(dummy_input))
