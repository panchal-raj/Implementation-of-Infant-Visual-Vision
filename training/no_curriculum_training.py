import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from src.dataloader import create_no_curriculum_dataloader
from training.train import train_model
from training.utils import plot_learning_curves
from models.resnet18Sohan import get_model
from config import EPOCHS, BATCH_SIZE, LEARNING_RATE, NUM_CLASSES, DEVICE, MODELS

# 1. Load Tiny ImageNet Data
def load_tiny_imagenet_data(split="train"):
    data = load_dataset("zh-plus/tiny-imagenet")
    return data[split]

# 2. No Curriculum Training Function
def train_and_save_model_no_curriculum(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, num_epochs, model_output_dir, loss_output_dir, model_name):
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(loss_output_dir, exist_ok=True)  

    # Train the model
    stage_name = f"No Curriculum - {model_name}"
    train_losses, val_losses = train_model(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, num_epochs, stage_name)

    # Save the final model
    torch.save(model.state_dict(), os.path.join(model_output_dir, f"{model_name}_no_curriculum_final.pth"))

    # Save losses
    loss_file = f"{model_name}_no_curriculum_losses.json"
    with open(os.path.join(loss_output_dir, loss_file), "w") as f:
        json.dump({"train_losses": train_losses, "val_losses": val_losses}, f)

    # Plot learning curves
    plot_learning_curves(train_losses, val_losses, stage_name)

    return train_losses, val_losses

# 3. Main Training Loop (No Curriculum)
def train_no_curriculum(batch_size, num_epochs, learning_rate, num_classes, model_output_dir, loss_output_dir, model_names):
    # 3.1 Load dataset
    train_data = load_tiny_imagenet_data(split="train")
    val_data = load_tiny_imagenet_data(split="valid")

    # 3.2 Create No curriculum DataLoaders
    train_dataloader = create_no_curriculum_dataloader(train_data, batch_size)
    val_dataloader = create_no_curriculum_dataloader(val_data, batch_size)

    # 3.3 Train each model
    for model_name in model_names:
        # Initialize model, criterion, optimizer, and scheduler
        model = get_model(model_name, num_classes=num_classes)
        device = torch.device("cuda" if torch.cuda.is_available() else DEVICE)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.95, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataloader) * num_epochs)

        # Train and save model
        train_and_save_model_no_curriculum(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, num_epochs, model_output_dir, loss_output_dir, model_name)

# 4. Run Script
def main():
    # Hyperparameters
    batch_size = BATCH_SIZE
    num_epochs = EPOCHS
    learning_rate = LEARNING_RATE
    num_classes = NUM_CLASSES  # Tiny ImageNet has 200 classes

    model_output_dir = "output/models/no_curriculum/"
    loss_output_dir = "output/loss_logs/no_curriculum/"
    #model_names = ["resnet18", "vgg16", "alexnet"]
    model_names = MODELS

    train_no_curriculum(batch_size, num_epochs, learning_rate, num_classes, model_output_dir, loss_output_dir, model_names)

if __name__ == "__main__":
    main()