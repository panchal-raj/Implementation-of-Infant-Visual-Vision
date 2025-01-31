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

# 1. Load Tiny ImageNet Data
def load_tiny_imagenet_data(split="train"):
    data = load_dataset("zh-plus/tiny-imagenet")
    return data[split]

# 2. Curriculum Training with Flexible Epoch Allocation
def train_and_save_visual_acuity_model(model,
                                    curriculum_dataloaders,
                                    val_dataloader,
                                    ages,
                                    stage_epochs,
                                    criterion,
                                    optimizer,
                                    scheduler,
                                    total_epochs,
                                    model_output_dir,
                                    loss_output_dir,
                                    model_name):
    """
    Train the model through multiple 'stages' corresponding to different infant ages.
    
    stage_epochs: A dictionary mapping each age to the number of epochs for that stage.
                  Example: {6: 3, 9: 5, 12: 7} sums to 15 total epochs.
    """
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(loss_output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else DEVICE)
    model = model.to(device)

    # Track overall losses across all stages
    overall_train_losses = []
    overall_val_losses = []

    # Loop through each age stage in the order given by `ages`
    for age in ages:
        stage_name = f"visual_acuity_Age_{age}mo_{model_name}"
        dataloader = curriculum_dataloaders[age]
        epochs_for_this_stage = stage_epochs[age]

        print(f"[{stage_name}] -> Training for {epochs_for_this_stage} epoch(s).")

        train_losses, val_losses = train_model(
            model=model,
            dataloader=dataloader,
            val_dataloader=val_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=epochs_for_this_stage,
            stage_name=stage_name
        )

        overall_train_losses.extend(train_losses)
        overall_val_losses.extend(val_losses)

    # Ensure we have the total epochs accounted for
    assert len(overall_train_losses) == total_epochs, (
        "The sum of stage epochs does not match the total_epochs."
    )

    # Save the final model
    torch.save(model.state_dict(), os.path.join(model_output_dir, f"{model_name}_visual_acuity_final.pth"))

    # Save the train/val losses to JSON
    loss_file = f"{model_name}_visual_acuity_losses.json"
    with open(os.path.join(loss_output_dir, loss_file), "w") as f:
        json.dump({"train_losses": overall_train_losses, "val_losses": overall_val_losses}, f)

    # Plot learning curves for the entire curriculum
    final_stage_name = f"VisualAcuity_{model_name}"
    plot_learning_curves(overall_train_losses, overall_val_losses, final_stage_name)

    return overall_train_losses, overall_val_losses

# 3. Helper Function: Create a default or custom stage-to-epoch mapping
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

# 4. Main Training Loop (Curriculum)
def train_visual_acuity(batch_size,
                     total_epochs,
                     learning_rate,
                     num_classes,
                     model_output_dir,
                     loss_output_dir,
                     model_names,
                     ages=AGES,
                     user_epoch_map=None):
    """
    - Loads Tiny ImageNet training and validation data.
    - Creates dataloaders with age-based transforms (visual acuity).
    - Creates and trains multiple models with the total epochs distributed
      according to a user-provided or default scheme.
    """
    # 4.1 Load dataset
    train_data = load_tiny_imagenet_data(split="train")
    val_data = load_tiny_imagenet_data(split="valid")

    # 4.2 Create curriculum DataLoaders (one DataLoader per age)
    curriculum_dataloaders = create_acuity_dataloader(
        dataset=train_data,
        ages=ages,
        batch_size=batch_size
    )

    # Single (no transform) validation DataLoader
    val_dataloader = create_no_curriculum_dataloader(val_data, batch_size)

    # 4.3 Train each model
    for model_name in model_names:
        # Create model
        model = get_model(model_name, num_classes=num_classes)

        # Define loss function, optimizer, scheduler
        device = torch.device("cuda" if torch.cuda.is_available() else DEVICE)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                              momentum=0.95, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

        # Create stage-epoch mapping (user or default)
        stage_epochs = create_stage_epoch_mapping(
            ages=ages,
            total_epochs=total_epochs,
            user_mapping=user_epoch_map
        )

        # Train and save model
        train_and_save_visual_acuity_model(
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
            model_name=model_name
        )


# 5. Example Entry Point
def main():
    # Hyperparameters
    batch_size = BATCH_SIZE
    total_epochs = EPOCHS
    learning_rate = LEARNING_RATE
    num_classes = NUM_CLASSES
    ages = AGES
    #model_names = ["resnet18", "vgg16", "alexnet"]
    model_names = MODELS


    # Output folders
    model_output_dir = "output/models/visual_acuity/"
    loss_output_dir = "output/loss_logs/visual_acuity/"

    # User-provided epoch distribution (optional)
    user_epoch_map = None  # Example: {6: 2, 9: 5, 12: 8}

    # Start training with curriculum for all models
    train_visual_acuity(
        batch_size=batch_size,
        total_epochs=total_epochs,
        learning_rate=learning_rate,
        num_classes=num_classes,
        model_output_dir=model_output_dir,
        loss_output_dir=loss_output_dir,
        model_names=model_names,
        ages=ages,
        user_epoch_map=user_epoch_map
    )

if __name__ == "__main__":
    main()

