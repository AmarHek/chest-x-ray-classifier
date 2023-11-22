from torch import nn
import torchvision.models as tvmodels
from params.model_params import ModelParams


class FoundationModel(nn.Module):

    def __init__(self, model_params: ModelParams):

        super(FoundationModel, self).__init__()
        self.model_params = model_params


