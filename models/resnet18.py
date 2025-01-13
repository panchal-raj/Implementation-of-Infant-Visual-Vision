# Sohan model
from torchvision.models import resnet18
from torch import nn

def get_resnet18(num_classes=200):
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model