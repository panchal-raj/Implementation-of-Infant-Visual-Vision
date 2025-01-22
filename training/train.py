import torch
import logging
from setting import DEVICE

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for detailed logs, INFO for general logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler("training.log", mode='w')  # Log to a file
    ]
)

def train_model(model, dataloader, val_dataloader, criterion, optimizer, scheduler, num_epochs, stage_name):
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else DEVICE)
    #device = torch.device("mps")
    logging.info(f"[{stage_name}] Training model on device: {device}")

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        logging.info(f"[{stage_name}] Starting Epoch {epoch + 1}/{num_epochs}")

        # Training phase
        model.train()
        running_loss = 0.0
        logging.info(f"[{stage_name}] Epoch {epoch + 1}: Starting training phase")
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, targets)
            logging.info(f"[{stage_name}] Epoch {epoch + 1}, Batch {batch_idx + 1}: Loss = {loss.item()}")

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(dataloader)
        train_losses.append(avg_train_loss)
        logging.info(f"[{stage_name}] Epoch {epoch + 1}: Training phase completed. Avg Train Loss: {avg_train_loss}")

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        logging.info(f"[{stage_name}] Epoch {epoch + 1}: Starting validation phase")
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_dataloader):
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                outputs = model(inputs)

                # Compute loss
                loss = criterion(outputs, targets)
                logging.debug(
                    f"[{stage_name}] Epoch {epoch + 1}, Validation Batch {batch_idx + 1}: Loss = {loss.item()}")

                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        logging.info(f"[{stage_name}] Epoch {epoch + 1}: Validation phase completed. Avg Val Loss: {avg_val_loss}")

        # Step the scheduler
        scheduler.step()
        logging.info(f"[{stage_name}] Epoch {epoch + 1}: Scheduler step completed")

        # Epoch summary
        logging.info(f"[{stage_name}] Epoch {epoch + 1} Summary: Train Loss = {avg_train_loss}, Val Loss = {avg_val_loss}")

    logging.info(f"[{stage_name}] Training completed for all {num_epochs} epochs.")
    return train_losses, val_losses
