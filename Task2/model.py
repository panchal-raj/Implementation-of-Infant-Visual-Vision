from imports import *

class CustomResNet(nn.Module):
    def __init__(self, model, num_classes):
        super(CustomResNet, self).__init__()
        # self.model = model
        model.conv1 = nn.Conv2d(
            in_channels=3, 
            out_channels=64, 
            kernel_size=5,  
            stride=2,       
            padding=1       
        )
        
        self.part = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.avgpool
        )
        
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 64, 64)
            feature_map = self.part(dummy_input)
            print("Feature map size after layer2:", feature_map.shape)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.part(x)         
        x = self.flatten(x)
        x = self.fc(x)
        return x

model = torchvision.models.resnet18(pretrained=False)
x = torch.randn(1, 3, 64, 64)
num_classes = 200
model = CustomResNet(model=model, num_classes=num_classes)
model(x).shape
print(model)
model = model.to(device)
