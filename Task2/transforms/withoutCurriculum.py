from imports import *
from model import model
from dataLoader import withoutTransforms

criterion = nn.CrossEntropyLoss()

# Optimizer: Adam
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-4)

# Scheduler (StepLR to reduce learning rate every 10 epochs)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# For storing loss and accuracy
train_losses, train_acc, val_losses, val_acc = [], [], [], []

# calling dataLoader without Transfromations
train_loaders , val_loaders = withoutTransforms()


# Best model parameters
best_model_path = 'v1_without_Curriculum.pth'

# Loop through age groups
for age_idx, age_group in enumerate(age_groups):
    print(f"\nStarting training for age group {age_group} months.")
    
    # Reset early stopping parameters for each age group
    patience = 5  # Number of epochs to wait for improvement
    counter = 0    # Counts epochs with no improvement
    best_val_loss = float('inf')  # Best validation loss

    # Loop through epochs for the current age group
    for epoch in range(32):  # 32 epochs per age group
        overall_epoch = epoch + (age_idx * 32) + 1  # Tracking overall epoch number
        print(f"\nEpoch {overall_epoch}/{num_epochs} (Age Group: {age_group} months, Epoch {epoch + 1}/32)")

        # Training phase
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        train_loader_tqdm = tqdm(train_loaders[age_idx], desc="Train Progress", leave=True)

        for inputs, labels in train_loader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Calculate training loss and accuracy
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct_train += predicted.eq(labels).sum().item()
            total_train += labels.size(0)

            # Update progress bar description with current metrics
            train_loader_tqdm.set_postfix(loss=loss.item())

        train_loss /= len(val_loaders[age_idx].dataset)
        train_accuracy = 100 * correct_train / total_train

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        val_loader_tqdm = tqdm(val_loaders[age_idx], desc="Validation Progress", leave=True)

        with torch.no_grad():
            for inputs, labels in val_loader_tqdm:
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Calculate validation loss and accuracy
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                correct_val += predicted.eq(labels).sum().item()
                total_val += labels.size(0)

                # Update progress bar description with current metrics
                val_loader_tqdm.set_postfix(loss=loss.item())

        val_loss /= len(val_loaders[age_idx].dataset)
        val_accuracy = 100 * correct_val / total_val

        # Update learning rate scheduler
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_acc.append(train_accuracy)
        val_acc.append(val_accuracy)
        
        print(f"train Loss: {train_loss:.4f}, train_acc: {train_accuracy:.2f}%, val_loss: {val_loss:.4f}, val_acc: {val_accuracy:.2f}%")

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0  # Reset the counter if validation loss improves
            torch.save(model.state_dict(), best_model_path)  # Save best model
            print(f"Validation loss improved. Saving model to {best_model_path}.")
        else:
            counter += 1
            print(f"No improvement in validation loss. Counter: {counter}/{patience}")
            if counter >= patience:
                print("Early stopping triggered for this age group!")
                break

    print(f"Training completed for age group {age_group} months. Best model saved to {best_model_path}.")

# Load the best model's weights
model.load_state_dict(torch.load(best_model_path))
print(f"Training complete. Best model loaded from {best_model_path}.")

# Determine completed epochs based on the collected metrics
completed_epochs = len(train_losses)  # Matches the length of collected metrics
epochs = np.arange(completed_epochs) + 1  # Generate array of epochs from 1 to completed_epochs

# Plotting accuracy and loss
plt.figure(figsize=(12, 6))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs, train_acc[:completed_epochs], label="Train Accuracy")
plt.plot(epochs, val_acc[:completed_epochs], label="Validation Accuracy")
plt.title("Train and Validation Accuracy without Curriculum")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(epochs, train_losses[:completed_epochs], label="Train Loss")
plt.plot(epochs, val_losses[:completed_epochs], label="Validation Loss")
plt.title("Train and Validation Loss without Curriculum")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()

# Save the plot after displaying it
plot_path = "without_Curriculum_epochs_plot.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
print(f"Plot saved to {plot_path}")