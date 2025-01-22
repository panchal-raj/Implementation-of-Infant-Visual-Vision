import torch
AGES = [3, 6, 9, 12]

## Project PART 2
EPOCHS = 6
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_CLASSES = 200  # Tiny ImageNet has 200 classes
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")
#MODELS = ["resnet18", "vgg16", "alexnet"]
MODELS = ["resnet18"]

MODEL_PATHS = {
    "color_perception": "./output/models/Sohan/resnet18_color_perception_final.pth",
    "curriculum": "./output/models/Sohan/resnet18_curriculum_final.pth",
    "no_curriculum": "./output/models/Sohan/resnet18_no_curriculum_final.pth",
    "visual_acuity": "./output/models/Sohan/resnet18_visual_acuity_final.pth"
}

LAYERS = ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool", "fc"]