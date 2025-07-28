import torch.nn as nn
from torchvision import models

def ResNet_Definition(num_classes):
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(nn.Linear(2048, num_classes))
    return model