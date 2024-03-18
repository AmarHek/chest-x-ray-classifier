from transformers import AutoImageProcessor, AutoModel, AutoModelForImageClassification, Dinov2ForImageClassification
import torch
from torch.utils.data import DataLoader
from src.datasets.chexpert import CheXpert
from src.params import CheXpertParams
import requests
import torchvision


def get_num_features(feature_extractor) -> int:
    dummy_input = torch.zeros(1, 3, 224, 224)
    # Run a forward pass of the dummy input tensor through the feature extractor
    with torch.no_grad():
        output = feature_extractor(dummy_input)
    # Return the channel dimension of the output
    return output.last_hidden_state.shape


dataParams = CheXpertParams()
dataParams.image_root_path = "/home/amar/Documents/"
dataParams.csv_path = "/home/amar/Documents/CheXpert-v1.0-small/valid.csv"
dataParams.augment = False

dataset = CheXpert(dataParams)
dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

model = AutoModel.from_pretrained("facebook/dinov2-base")

model.eval()

print(model)

batch = next(iter(dataloader))
images = batch['image']

with torch.no_grad():
    outputs = model(images)
    print(outputs)

print(get_num_features(model))

print(torchvision.models.swin_v2_b())
print(AutoModelForImageClassification.from_pretrained("microsoft/swinv2-base-patch4-window16-256"))