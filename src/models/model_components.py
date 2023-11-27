import torchvision.models
import torch.nn as nn
from torch.nn import Sequential


def get_pretrained_model(architecture: str, weights: str = "DEFAULT") -> nn.Module:
    architectures = {
        "densenet121": torchvision.models.densenet121,
        "densenet161": torchvision.models.densenet161,
        "efficientnet_b1": torchvision.models.efficientnet_b1,
        "efficientnet_b2": torchvision.models.efficientnet_b2,
        "efficientnet_v2_s": torchvision.models.efficientnet_v2_s,
        "efficientnet_v2_m": torchvision.models.efficientnet_v2_m,
        "efficientnet_v2_l": torchvision.models.efficientnet_v2_l,
        "resnet50": torchvision.models.resnet50,
        "resnet101": torchvision.models.resnet101,
        "resnet152": torchvision.models.resnet152,
        "resnext50_32x4d": torchvision.models.resnext50_32x4d,
        "resnext101_32x8d": torchvision.models.resnext101_32x8d,
        "resnext101_64x4d": torchvision.models.resnext101_64x4d
    }

    architecture = architecture.lower()

    assert architecture in architectures.keys(), "%s is an invalid architecture!" % architecture

    if pretrained:
        weights = weights_dict[architecture]
    else:
        weights = None

    return architectures[architecture](weights=weights)


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


def get_classifier(architecture: str, n_classes: int, classifier_function,
                   long_classifier: bool = False, dropout: float = 0.2) -> nn.Module:

    shape_dict = {
        "densenet121": 1024,
        "densenet161": 2208,
        "efficientnet_b1": 1280,
        "efficientnet_b2": 1408,
        "efficientnet_v2_s": 1280,
        "efficientnet_v2_m": 1280,
        "efficientnet_v2_l": 1280,
        "resnet50": 2048,
        "resnet101": 2048,
        "resnet152": 2048,
        "resnext50_32x4d": 2048,
        "resnext101_32x8d": 2048,
        "resnext101_64x4d": 2048
    }

    assert architecture in shape_dict.keys(), "Invalid architecture!"
    out_features = shape_dict[architecture]

    # define a new classifier with custom layers
    if long_classifier:
        classifier = nn.Sequential(nn.Linear(out_features, 512),
                                   nn.ReLU(),
                                   nn.Dropout(dropout),
                                   nn.Linear(512, n_classes),
                                   classifier_function)
    else:
        classifier = nn.Sequential(nn.Linear(out_features, n_classes),
                                   classifier_function)

    return classifier


def sequential_model(architecture: str, n_classes: int, pretrained: bool = True, finetune: bool = False,
                     long_classifier: bool = False, classifier_function: str = "sigmoid",
                     dropout: float = 0.2) -> nn.Module:

    # get pretrained model and classifier function
    model = get_pretrained_model(architecture, pretrained=pretrained)
    classifier_function = get_classifier_function(classifier_function)

    # freeze all pretrained layers if wanted
    if finetune:
        for param in model.parameters():
            param.requires_grad = False

    classifier = get_classifier(architecture, n_classes, classifier_function=classifier_function,
                                long_classifier=long_classifier, dropout=dropout)

    # replace the pretrained model's classifier with our new one
    if "resnet" in architecture or "resnext" in architecture:
        model.fc = classifier
    else:
        model.classifier = classifier

    return model
