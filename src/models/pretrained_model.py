import torch
from torch import nn
#import torch.nn.utils.prune as prune
#import torch.nn.functional as F

from params import PretrainedModelParams
from models import get_backbone, freeze_layers, classifier_head, classifier_function




class PretrainedModel(nn.Module):

    def __init__(self, model_params: PretrainedModelParams):

        super(PretrainedModel, self).__init__()
        self.model_params = model_params

        self.backbone = get_backbone(self.model_params.backbone, self.model_params.weights)
        freeze_layers(self.backbone, self.model_params.freeze_depth)

        num_features = self.get_num_features(self.backbone)

        avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        if self.model_params.dropout > 0:
            dropout = nn.Dropout(self.model_params.dropout)
        else:
            dropout = nn.Identity()

        head = classifier_head(in_features=num_features, **self.model_params.to_dict())
        cls_function = classifier_function(self.model_params.classifier_function)
        self.classifier = nn.Sequential(avg_pool, dropout, head, cls_function)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

    @staticmethod
    def get_num_features(feature_extractor: nn.Module) -> int:
        dummy_input = torch.zeros(1, 3, 224, 224)
        # Run a forward pass of the dummy input tensor through the feature extractor
        with torch.no_grad():
            output = feature_extractor(dummy_input)
        # Return the channel dimension of the output
        return output.shape[1]


if __name__ == "__main__":
   
   dummy_input = torch.zeros(4, 3, 224, 224)

   model_params = PretrainedModelParams()
   model_params.backbone = "efficientnet_b0"
   model_params.head = "csra"
   model_params.num_heads = 4
   model_params.lam = 0.1
   model_params.num_classes = 14
   model = PretrainedModel(model_params)
# Step 1: Define your model architecture
# For example, let's use a pre-trained ResNet model
#model = models.densenet121(pretrained=False)

   #model_path = 'Baseline_2024-01-26_19-38-22/Baseline_2024-01-26_19_38_22_2.pth'


# Step 2: Load the pre-trained weights from the .pth file
   #file_path = f'/autofs/home/stud/jaeger/chest-x-ray-classifier/experiments/{model_path}'

   #fp = "/autofs/stud/jaeger/chest-x-ray-classifier/experiments/Baseline_2024-01-26_19-38-22/Baseline_2024-01-26_19-38-22_2.pth"
   #checkpoint = torch.load(fp, map_location=torch.device('cpu'))
##print(list(checkpoint.keys()))
#print(checkpoint['model'].weights)
#print(checkpoint['model'])
# If the .pth file contains the state_dict() only, use the foll>#model.load_state_dict(checkpoint)

# If the .pth file contains more than just the state_dict (e.g.>#model.load_state_dict(checkpoint['modelParams'])

   #model_params = checkpoint['modelParams']

   
   #print(model.backbone.conv0)
   
   #module pruning
   #module = model.backbone.conv0
   #prune.random_unstructured(module, name="weight", amount=0.3)
   #print(list(module.named_buffers()))
   #print('successfully ran through')
   print(model(dummy_input).shape)
