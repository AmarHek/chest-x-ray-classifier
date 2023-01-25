import torchvision.models
import torch.nn as nn


def get_pretrained_model(architecture: str, pretrained: bool = True) -> nn.Module:
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
        "resnext101_64x4d": torchvision.models.resnext101_64x4d,
        "inception_v3": torchvision.models.inception_v3
    }

    architecture = architecture.lower()

    assert architecture in architectures.keys(), "%s is an invalid architecture!" % architecture

    return architectures[architecture](pretrained=pretrained)


def get_classifier_function(classifier_function: str):
    classifier_functions = {
        "sigmoid": nn.Sigmoid(),
        "softmax": nn.Softmax(),
        "logsoftmax": nn.LogSoftmax(),
        "logsigmoid": nn.LogSigmoid()
    }

    classifier_function = classifier_function.lower()

    assert classifier_function in classifier_functions.keys(), \
        "%s is an invalid classifier function!" % classifier_function

    return classifier_functions[classifier_function]


def sequential_model(architecture: str, n_classes: int, finetune: bool = False, long_classifier: bool = False,
                     classifier_function: str = "sigmoid", dropout: float = 0.2) -> nn.Module:

    # get pretrained model and classifier function
    model = get_pretrained_model(architecture)
    print(model)
    classifier_function = get_classifier_function(classifier_function)

    # freeze all pretrained layers if wanted
    if finetune:
        for param in model.parameters():
            param.requires_grad = False

    # define a new classifier with custom layers
    if long_classifier:
        classifier = nn.Sequential(nn.LazyLinear(512),
                                   nn.ReLU(),
                                   nn.Dropout(dropout),
                                   nn.Linear(512, n_classes),
                                   classifier_function)
    else:
        classifier = nn.Sequential(nn.LazyLinear(n_classes),
                                   classifier_function)

    # replace the pretrained model's classifier with our new one
    if "resnset" in architecture or "inception" in architecture:
        model.fc = classifier
    else:
        model.classifier = classifier

    return model


if __name__ == "__main__":
    architecture = "inception_v3"
    classifier_function = "Sigmoid"
    n_classes = 5

    model = sequential_model(architecture, n_classes, classifier_function=classifier_function)
    print(model)
