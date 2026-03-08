import torch
import torch.nn as nn
from torchvision import models


def get_model(num_classes):

    from torchvision.models import resnet50, ResNet50_Weights

    model = resnet50(weights=ResNet50_Weights.DEFAULT)

    # replace final layer
    in_features = model.fc.in_features

    model.fc = nn.Linear(in_features, num_classes)

    return model