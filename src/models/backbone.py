import torch
import torchvision.models
import torch.nn as nn
from torch.nn import Sequential

from models.util import Unsqueeze

# all available architectures
architectures = {
    "densenet121": torchvision.models.densenet121,
    "densenet161": torchvision.models.densenet161,
    "densenet169": torchvision.models.densenet169,
    "densenet201": torchvision.models.densenet201,
    "efficientnet_b0": torchvision.models.efficientnet_b0,
    "efficientnet_b1": torchvision.models.efficientnet_b1,
    "efficientnet_b2": torchvision.models.efficientnet_b2,
    "efficientnet_b3": torchvision.models.efficientnet_b3,
    "efficientnet_b4": torchvision.models.efficientnet_b4,
    "efficientnet_b5": torchvision.models.efficientnet_b5,
    "efficientnet_b6": torchvision.models.efficientnet_b6,
    "efficientnet_b7": torchvision.models.efficientnet_b7,
    "efficientnet_v2_s": torchvision.models.efficientnet_v2_s,
    "efficientnet_v2_m": torchvision.models.efficientnet_v2_m,
    "efficientnet_v2_l": torchvision.models.efficientnet_v2_l,
    "resnet18": torchvision.models.resnet18,
    "resnet50": torchvision.models.resnet50,
    "resnet101": torchvision.models.resnet101,
    "resnet152": torchvision.models.resnet152,
    "resnext50_32x4d": torchvision.models.resnext50_32x4d,
    "resnext101_32x8d": torchvision.models.resnext101_32x8d,
    "resnext101_64x4d": torchvision.models.resnext101_64x4d,
    "vit_b_16": torchvision.models.vit_b_16,
    "vit_b_32": torchvision.models.vit_b_32,
    "vit_l_16": torchvision.models.vit_l_16,
    "vit_l_32": torchvision.models.vit_l_32,
    "vit_h_14": torchvision.models.vit_h_14,
    "swin_b": torchvision.models.swin_b,
    "swin_t": torchvision.models.swin_t,
    "swin_s": torchvision.models.swin_s,
    "swin_v2_b": torchvision.models.swin_v2_b,
    "swin_v2_t": torchvision.models.swin_v2_t,
    "swin_v2_s": torchvision.models.swin_v2_s,
    "vgg11": torchvision.models.vgg11
}


def get_backbone(architecture: str, weights: str = "DEFAULT") -> nn.Module:

    architecture = architecture.lower()
    assert architecture in architectures.keys(), "%s is an invalid architecture!" % architecture

    # load the model
    backbone = architectures[architecture](weights=weights)

    # preprocess based on architecture
    if any(model_base in architecture for model_base in ["efficientnet", "densenet", "vgg11"]):
        backbone = backbone.features
    elif "swin" in architecture:
        backbone = Sequential(*list(backbone.children())[:-3])
    elif "vit" in architecture:
        # for vit, we need to add height and width to the output
        # since we want to remove heads anyway, we can just replace it with our
        # Unsqueeze for adding height and width dimensions
        backbone.heads = Unsqueeze(dim=2, n_dims=2)
    elif "inception" in architecture:
        backbone = Sequential(*list(backbone.children())[:-3])
    elif "resnet" in architecture or "resnext" in architecture:
        backbone = Sequential(*list(backbone.children())[:-2])
    else:
        raise ValueError("Unsupported model architecture. Please modify the code accordingly.")

    return backbone


def freeze_layers(model: nn.Module, depth: int = -1):
    """
    Freeze layers of a model up to a certain depth
    :param model: the model to freeze layers of
    :param depth: the depth up to which to freeze layers, -1 = freeze no layers, 0 = freeze all layers,
                  1 = freeze all but last layer, etc.
    :return: None
    """
    if depth == -1:
        # Unfreeze all layers
        for param in model.parameters():
            param.requires_grad = True
    elif depth == 0:
        # Freeze all layers
        for param in model.parameters():
            param.requires_grad = False
    elif 0 < depth <= len(list(model.children())):
        # Freeze layers up to freeze_depth (excluding)
        for i, child in enumerate(model.children()):
            freeze_layers(child, depth - 1 if i == 0 else depth)
    else:
        raise ValueError(f"Invalid freeze_depth value. It should be between -1 and {len(list(model.children()))}.")
