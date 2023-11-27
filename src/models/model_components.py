import torchvision.models
import torch.nn as nn
from torch.nn import Sequential


def get_backbone(architecture: str, weights: str = "DEFAULT") -> nn.Module:

    # all available architectures
    architectures = {
        "densenet121": torchvision.models.densenet121,
        "densenet161": torchvision.models.densenet161,
        "densenet169": torchvision.models.densenet169,
        "densenet201": torchvision.models.densenet201,
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
        "resnet50": torchvision.models.resnet50,
        "resnet101": torchvision.models.resnet101,
        "resnet152": torchvision.models.resnet152,
        "resnext50_32x4d": torchvision.models.resnext50_32x4d,
        "resnext101_32x8d": torchvision.models.resnext101_32x8d,
        "resnext101_64x4d": torchvision.models.resnext101_64x4d,
        "swin_b": torchvision.models.swin_b,
        "swin_t": torchvision.models.swin_t,
        "swin_s": torchvision.models.swin_s,
        "swin_v2_b": torchvision.models.swin_v2_b,
        "swin_v2_t": torchvision.models.swin_v2_t,
        "swin_v2_s": torchvision.models.swin_v2_s,
        "inception_v3": torchvision.models.inception_v3,
        "vgg11": torchvision.models.vgg11
    }

    architecture = architecture.lower()
    assert architecture in architectures.keys(), "%s is an invalid architecture!" % architecture

    # load the model
    backbone = architectures[architecture](weights=weights)

    # preprocess based on architecture
    if any(model_base in architecture for model_base in ["efficientnet", "swin", "densenet", "vgg11"]):
        backbone = backbone.features
    elif "inception" in architecture:
        backbone = Sequential(*list(backbone.children())[:-3])
    elif "resnet" in architecture or "resnext" in architecture:
        backbone = Sequential(*list(backbone.children())[:-2])
    else:
        raise ValueError("Unsupported model architecture. Please modify the code accordingly.")

    return backbone


def get_classifier_function(classifier_function: str):
    classifier_functions = {
        "none": nn.Identity(),
        "sigmoid": nn.Sigmoid(),
        "softmax": nn.Softmax(),
        "logsoftmax": nn.LogSoftmax(),
        "logsigmoid": nn.LogSigmoid()
    }

    classifier_function = classifier_function.lower()

    assert classifier_function in classifier_functions.keys(), \
        "%s is an invalid classifier function!" % classifier_function

    return classifier_functions[classifier_function]
