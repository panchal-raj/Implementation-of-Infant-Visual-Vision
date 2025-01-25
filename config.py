import torch
AGES = [3, 6, 9, 12]

## Project PART 2
EPOCHS = 6
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_CLASSES = 200  # Tiny ImageNet has 200 classes
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")

MODEL_PATHS = {
    "visual_acuity": "./weights/FabianResNet/resnet18_tinyimagenet_acuity.pth",
    "contrast_adjust": "./weights/FabianResNet/resnet18_tinyimagenet_contrast.pth",
    "both_transforms": "./weights/FabianResNet/resnet18_tinyimagenet_both.pth",
    "no_transformation": "./weights/FabianResNet/resnet18_tinyimagenet_default.pth"
}

