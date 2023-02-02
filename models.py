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

    weights_dict = {
        "densenet121": torchvision.models.DenseNet121_Weights.DEFAULT,
        "densenet161": torchvision.models.DenseNet161_Weights.DEFAULT,
        "efficientnet_b1": torchvision.models.EfficientNet_B1_Weights.DEFAULT,
        "efficientnet_b2": torchvision.models.EfficientNet_B2_Weights.DEFAULT,
        "efficientnet_v2_s": torchvision.models.EfficientNet_V2_S_Weights.DEFAULT,
        "efficientnet_v2_m": torchvision.models.EfficientNet_V2_M_Weights.DEFAULT,
        "efficientnet_v2_l": torchvision.models.EfficientNet_V2_L_Weights.DEFAULT,
        "resnet50": torchvision.models.ResNet50_Weights.DEFAULT,
        "resnet101": torchvision.models.ResNet101_Weights.DEFAULT,
        "resnet152": torchvision.models.ResNet152_Weights.DEFAULT,
        "resnext50_32x4d": torchvision.models.ResNeXt50_32X4D_Weights.DEFAULT,
        "resnext101_32x8d": torchvision.models.ResNeXt101_32X8D_Weights.DEFAULT,
        "resnext101_64x4d": torchvision.models.ResNeXt101_64X4D_Weights.DEFAULT,
        "inception_v3": torchvision.models.Inception_V3_Weights.DEFAULT
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
        "sigmoid": nn.Sigmoid(),
        "softmax": nn.Softmax(),
        "logsoftmax": nn.LogSoftmax(),
        "logsigmoid": nn.LogSigmoid()
    }

    classifier_function = classifier_function.lower()

    assert classifier_function in classifier_functions.keys(), \
        "%s is an invalid classifier function!" % classifier_function

    return classifier_functions[classifier_function]


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
    if "resnet" in architecture or "inception" in architecture:
        model.fc = classifier
    else:
        model.classifier = classifier

    return model


if __name__ == "__main__":
    model = sequential_model("inception_v3", 5, classifier_function="sigmoid")
    print(model)
