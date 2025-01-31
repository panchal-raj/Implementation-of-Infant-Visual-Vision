from matplotlib import pyplot as plt
import os
import json


def plot_learning_curves(train_losses, val_losses, stage_name):
    output_dir = 'outputs/figures'
    os.makedirs(output_dir, exist_ok=True)

    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.title(f"Learning Curves ({stage_name})")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Across All Stages')
    plt.legend()
    plt.savefig(f'{output_dir}/{stage_name}_learning_curves.png')
    plt.close()
    
def load_loss_logs(log_dir):
    losses = {}
    for file_name in os.listdir(log_dir):
        if file_name.endswith(".json"):
            with open(os.path.join(log_dir, file_name), "r") as f:
                data = json.load(f)
                losses[file_name.replace(".json", "")] = data
    return losses

def plot_comparative_losses(loss_logs, output_path):
    output_dir = 'outputs/final_plot'
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    for scenario, loss_data in loss_logs.items():
        train_losses = loss_data["train_losses"]
        val_losses = loss_data["val_losses"]
        plt.plot(train_losses, label=f"{scenario} - Train")
        plt.plot(val_losses, label=f"{scenario} - Validation")
    
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Learning Curves Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/learning_curves.png')
    plt.close()

if __name__ == "__main__":
    # Specify directories and output paths
    loss_logs_dir = "loss_logs"
    comparison_plot_path = "learning_curves_comparison.png"
    
    # Load loss logs and plot
    loss_logs = load_loss_logs(loss_logs_dir)
    plot_comparative_losses(loss_logs, comparison_plot_path)