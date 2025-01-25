# Fabian model
from torchvision.models import resnet18
from torch import nn

def get_model(num_classes=200):
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Remove maxpool
    model.fc = nn.Sequential(
        nn.Dropout(0.5),  # Dropout to avoid overfitting
        nn.Linear(model.fc.in_features, num_classes)
    )
    return model

