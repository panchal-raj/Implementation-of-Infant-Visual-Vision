import torch
AGES = [3, 6, 9, 12]

## Project PART 2
EPOCHS = 6
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_CLASSES = 200  # Tiny ImageNet has 200 classes
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")

#Fabian Model 
MODEL_PATHS1 = {
    "visual_acuity": "./weights/FabianResNet/resnet18_tinyimagenet_acuity.pth",
    "contrast_adjust": "./weights/FabianResNet/resnet18_tinyimagenet_contrast.pth",
    "both_transforms": "./weights/FabianResNet/resnet18_tinyimagenet_both.pth",
    "no_transformation": "./weights/FabianResNet/resnet18_tinyimagenet_default.pth"
}

#Layer-Wise Model
MODEL_PATHS2 = {
    "visual_acuity": "./weights/LayerWiseResNet/resnet18_layerwise_acuity_final.pth",
    "color_adjust": "./weights/LayerWiseResNet/resnet18_layerwise_color_final.pth",
    "both_transforms": "./weights/LayerWiseResNet/resnet18_curriculum_final.pth",
    "no_transformation": "./weights/LayerWiseResNet/resnet18_no_curriculum_final.pth"
}


