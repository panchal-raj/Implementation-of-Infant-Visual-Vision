import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from src.dataloader import create_acuity_dataloader, create_no_curriculum_dataloader
from training.train import train_model
from training.utils import plot_learning_curves
from models.resnet18Sohan import get_model
from config import AGES, EPOCHS, BATCH_SIZE, LEARNING_RATE, NUM_CLASSES, DEVICE, MODELS


def load_tiny_imagenet_data(split="train"):
    """Load Tiny ImageNet data."""
    data = load_dataset("zh-plus/tiny-imagenet")
    return data[split]


def create_stage_epoch_mapping(ages, total_epochs, user_mapping=None):
    """
    Returns a dictionary that maps each age to a specific number of epochs.
    If user_mapping is provided, use that directly (and check correctness).
    Otherwise, split total_epochs equally across ages.
    """
    if user_mapping is not None:
        # Validate sum
        if sum(user_mapping.values()) != total_epochs:
            raise ValueError(
                "Sum of user-provided stage epochs != total_epochs. "
                f"Sum = {sum(user_mapping.values())}, total_epochs = {total_epochs}"
            )
        # Validate that all ages in 'ages' are in user_mapping
        for age in ages:
            if age not in user_mapping:
                raise ValueError(f"Missing stage epoch allocation for age {age}")
        return user_mapping
    else:
        # Default: evenly split total_epochs among the ages
        n_stages = len(ages)
        base_epochs = total_epochs // n_stages
        remainder = total_epochs % n_stages

        stage_epoch_map = {}
        for idx, age in enumerate(ages):
            # Distribute any remainder (1 extra epoch) among the first 'remainder' stages
            extra = 1 if idx < remainder else 0
            stage_epoch_map[age] = base_epochs + extra

        return stage_epoch_map


def train_layerwise_acuity_curriculum(model, curriculum_dataloaders, val_dataloader, ages, stage_epochs, criterion, optimizer, scheduler, total_epochs, model_output_dir, loss_output_dir, model_name):
    """Train the model using layer-wise curriculum learning."""
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(loss_output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else DEVICE)
    model = model.to(device)

    overall_train_losses = []
    overall_val_losses = []

    # Freeze all layers initially
    for param in model.parameters():
        param.requires_grad = False

    # Loop through each stage (age)
    for age in ages:
        stage_name = f"LayerWise_Acuity_Age_{age}mo_{model_name}"
        dataloader = curriculum_dataloaders[age]
        epochs_for_this_stage = stage_epochs[age]

        # Convert model.parameters() to a list for indexing
        params = list(model.parameters())

        # Unfreeze a proportion of parameters based on the age
        for param in params[:int(len(params) * (age / max(ages)))]:
            param.requires_grad = True

        print(f"[{stage_name}] -> Training for {epochs_for_this_stage} epoch(s).")

        # Train the model for this stage
        train_losses, val_losses = train_model(
            model=model,
            dataloader=dataloader,
            val_dataloader=val_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=epochs_for_this_stage,
            stage_name=stage_name,
        )

        overall_train_losses.extend(train_losses)
        overall_val_losses.extend(val_losses)

    # Save the model
    torch.save(model.state_dict(), os.path.join(model_output_dir, f"{model_name}_layerwise_acuity_final.pth"))

    # Save losses
    loss_file = f"{model_name}_layerwise_acuity_losses.json"
    with open(os.path.join(loss_output_dir, loss_file), "w") as f:
        json.dump({"train_losses": overall_train_losses, "val_losses": overall_val_losses}, f)

    # Plot learning curves
    final_stage_name = f"LayerWiseAcuityCurriculum_{model_name}"
    plot_learning_curves(overall_train_losses, overall_val_losses, final_stage_name)

    return overall_train_losses, overall_val_losses


def train_layerwise(batch_size, total_epochs, learning_rate, num_classes, model_output_dir, loss_output_dir, model_names, ages=AGES, user_epoch_map=None):
    """Main function for layer-wise curriculum learning."""
    # Load dataset
    train_data = load_tiny_imagenet_data(split="train")
    val_data = load_tiny_imagenet_data(split="valid")

    # Create curriculum dataloaders
    curriculum_dataloaders = create_acuity_dataloader(train_data, ages, batch_size)
    val_dataloader = create_no_curriculum_dataloader(val_data, batch_size)

    for model_name in model_names:
        # Initialize model
        model = get_model(model_name, num_classes)

        # Set up loss, optimizer, scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.95, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

        # Create stage-to-epoch mapping
        stage_epochs = create_stage_epoch_mapping(ages, total_epochs, user_epoch_map)

        # Train and save model
        train_layerwise_acuity_curriculum(
            model=model,
            curriculum_dataloaders=curriculum_dataloaders,
            val_dataloader=val_dataloader,
            ages=ages,
            stage_epochs=stage_epochs,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            total_epochs=total_epochs,
            model_output_dir=model_output_dir,
            loss_output_dir=loss_output_dir,
            model_name=model_name,
        )


def main():
    """Entry point for layer-wise curriculum learning."""
    batch_size = BATCH_SIZE
    total_epochs = EPOCHS
    learning_rate = LEARNING_RATE
    num_classes = NUM_CLASSES
    model_names = MODELS
    ages = AGES

    model_output_dir = "output/models/layerwise_acuity/"
    loss_output_dir = "output/loss_logs/layerwise_acuity/"

    # User-defined epoch mapping (optional)
    user_epoch_map = None  # Example: {6: 3, 9: 5, 12: 7}

    train_layerwise(batch_size, total_epochs, learning_rate, num_classes, model_output_dir, loss_output_dir, model_names, ages, user_epoch_map)


if __name__ == "__main__":
    main()
