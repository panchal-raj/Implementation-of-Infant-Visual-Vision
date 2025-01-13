import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from datasets import load_dataset

# Custom imports
from src.dataloader import create_no_curriculum_dataloader
from models.customModel import ImprovedVisionNet
from models.customResnet import CustomResNet
from models.resnet import get_resnet18
from config import DEVICE, BATCH_SIZE, EPOCHS, LEARNING_RATE, NUM_CLASSES

# Define output directories
MODEL_OUTPUT_DIR = "output/models"
LOSS_OUTPUT_DIR = "output/loss"
FIGURE_OUTPUT_DIR = "output/figure"
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(LOSS_OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_OUTPUT_DIR, exist_ok=True)

# Load Tiny ImageNet dataset from Hugging Face
def load_tiny_imagenet_data(split="train"):
    data = load_dataset("zh-plus/tiny-imagenet")
    return data[split]

# Load train and validation datasets
train_data = load_tiny_imagenet_data(split="train")
val_data = load_tiny_imagenet_data(split="valid")

# Function to train and validate the model
def train_and_validate_model(model, train_dataloader, val_dataloader, criterion, optimizer, total_epochs, model_name):
    model = model.to(DEVICE)
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(total_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in tqdm(train_dataloader, leave=False):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

        train_loss /= len(train_dataloader.dataset)
        train_accuracy = correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()

        val_loss /= len(val_dataloader.dataset)
        val_accuracy = correct_val / total_val
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch [{epoch+1}/{total_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    return train_losses, train_accuracies, val_losses, val_accuracies

# Function to save the model
def save_model(model, model_name):
    model_path = os.path.join(MODEL_OUTPUT_DIR, f"{model_name}/no_curriculum")
    os.makedirs(model_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_path, f"{model_name}_no_curriculum.pth"))
    print(f"Model saved to {model_path}")

# Function to plot and save training and validation learning curves
def plot_and_save(train_losses, train_accuracies, val_losses, val_accuracies, model_name):
    plt.figure(figsize=(12, 6))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve for {model_name} (No Curriculum)")
    plt.legend()

    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy Curve for {model_name} (No Curriculum)")
    plt.legend()

    figure_path = os.path.join(FIGURE_OUTPUT_DIR, f"{model_name}_no_curriculum.png")
    plt.savefig(figure_path)
    print(f"Figure saved to {figure_path}")
    plt.close()

# Main training logic
def main():
    print("Loading Tiny ImageNet dataset...")
    
    # Create dataloaders for train and validation sets
    train_dataloader = create_no_curriculum_dataloader(train_data, BATCH_SIZE)
    val_dataloader = create_no_curriculum_dataloader(val_data, BATCH_SIZE)

    # Models to train
    models = {
        "ImprovedVisionNet": ImprovedVisionNet(num_classes=NUM_CLASSES),
        "CustomResNet": CustomResNet(get_resnet18(), num_classes=NUM_CLASSES),
        "ResNet18": get_resnet18(num_classes=NUM_CLASSES)
    }

    for model_name, model in models.items():
        print(f"\nTraining {model_name} without curriculum learning...")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        train_losses, train_accuracies, val_losses, val_accuracies = train_and_validate_model(
            model, train_dataloader, val_dataloader, criterion, optimizer, EPOCHS, model_name
        )
        
        # Save model
        save_model(model, model_name)
        
        # Save training and validation results
        loss_path = os.path.join(LOSS_OUTPUT_DIR, f"{model_name}_no_curriculum_loss.pth")
        torch.save({"train_loss": train_losses, "train_accuracy": train_accuracies,
                    "val_loss": val_losses, "val_accuracy": val_accuracies}, loss_path)
        print(f"Loss saved to {loss_path}")
        
        # Plot and save learning curves
        plot_and_save(train_losses, train_accuracies, val_losses, val_accuracies, model_name)

if __name__ == "__main__":
    main()
