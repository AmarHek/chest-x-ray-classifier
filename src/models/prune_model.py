import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

from params import PretrainedModelParams
from models import get_backbone, freeze_layers, classifier_head, classifier_function

def prune_densenet(model, pruning_percentage):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=pruning_percentage)


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
    #dummy_input = torch.zeros(4, 3, 224, 224)

    #model_params = PretrainedModelParams()
    #model_params.backbone = "efficientnet_b0"
    #model_params.head = "csra"
    #model_params.num_heads = 4
    #model_params.lam = 0.1
    #model_params.num_classes = 14
                        
# Step 1: Define your model architecture
# For example, let's use a pre-trained ResNet model
#model = models.densenet121(pretrained=False)

   model_path = 'Baseline_2024-01-26_19-38-22/Baseline_2024-01-26_19_38_22_2.pth'


# Step 2: Load the pre-trained weights from the .pth file
   file_path = f'/autofs/home/stud/jaeger/chest-x-ray-classifier/experiments/{model_path}'

   fp = "/autofs/stud/jaeger/chest-x-ray-classifier/experiments/Baseline_2024-01-26_19-38-22/Baseline_2024-01-26_19-38-22_2.pth"
   checkpoint = torch.load(fp, map_location=torch.device('cpu'))
##print(list(checkpoint.keys()))
#print(checkpoint['model'].weights)
#print(checkpoint['model'])
# If the .pth file contains the state_dict() only, use the foll>#model.load_state_dict(checkpoint)

# If the .pth file contains more than just the state_dict (e.g.>#model.load_state_dict(checkpoint['modelParams'])

   model_params = checkpoint['modelParams']

   model = PretrainedModel(model_params)
   #print(model.backbone.conv0)
   
   #module pruning
   #module = model.backbone.conv0
   #prune.random_unstructured(module, name="weight", amount=0.3)
   #print(list(module.named_buffers()))
   modules = list(model.backbone)
   print(f"found {len(modules)} modules")
   counter = 0
   #for m in modules:
   #   print(type(m))
   #   layers = [a for a in dir(m) if not a.startswith('__') and not callable(getattr(m, a))]
   #   try:
   #         #print(m._modules)
   #         mods = list(m._modules)
   #         for mod in mods:
   #            print(counter)
   #            print(type(mod))
   #            print(mod)
   #            prune.random_unstructured(mod, name="weight", amount=0.3)
   #            
   #            counter+=1
   #   except:
   #            print("error")
   #            pass
      #print(layers,"\n\n\n")
#print(modules)
   pruning_method = prune.L1Unstructured(amount=0.2)  # 20% pruning
   # Apply pruning to the entire model
#   prune.global_unstructured(
#   parameters=model.backbone.parameters(),
#   pruning_method=pruning_method,
#)
   #print(type(model.super()))
   #print(type(model.backbone))
   #print(model.backbone.parameters())
   #print(model.parameters)
   for m in model.backbone.parameters():
      try:
         prune.l1_unstructured(m,name="weight", amount=0.8)
         print('successfully pruned')
      except:
                print('pruning error')
#print(list(model.backbone.parameters())[0])
   print(model.backbone.named_modules)
   pruning_percentage = .5
   prune_densenet(model.backbone, pruning_percentage)
   print('pruned model')
   #torch.save(model.backbone.state_dict(), f'pruned_densenet{int(pruning_percentage*100)}.pth')
   checkpoint['model'] = model.backbone.state_dict()
   torch.save(checkpoint, 'pruned_50.pth')
   print('successfully ran through')
   #print(model(dummy_input).shape)
   #prune.global_unstructured(
   #parameters=model.backbone.parameters(),
   #pruning_method=pruning_method,
   #)
