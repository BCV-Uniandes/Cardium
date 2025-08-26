import torch
from tqdm import tqdm
import wandb
import numpy as np
from torch.utils.data import DataLoader,  WeightedRandomSampler
import matplotlib.pyplot as plt
import os

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    device: torch.device,
    fold: int,
    args,
) -> np.ndarray:
    """
    Train the model for one epoch.

    Args:
        model: PyTorch model.
        train_loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        epoch: Current epoch number.
        device: Torch device (CPU/GPU).
        fold: Current fold index for logging.
        args: Additional arguments (e.g., num_epochs).

    Returns:
        logits (np.ndarray): Concatenated predictions for the entire epoch.
    """
    model.train()
    running_loss = 0.0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}", unit="batch")

    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        img_data, tab_data = inputs[0].to(device), inputs[1].to(device)
        targets = targets.to(device).float()  # Ensure float for BCE/ regression

        optimizer.zero_grad()

        # Forward pass
        outputs = model(img_data, tab_data)
        # Squeeze outputs
        outputs = outputs.squeeze(1)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Update progress bar with running loss
        progress_bar.set_postfix({
            "Batch Loss": f"{loss.item():.4f}",
            "Avg Loss": f"{running_loss / (batch_idx + 1):.4f}"
        })

    # Compute average loss for the epoch
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{args.num_epochs}], Average Loss: {avg_loss:.4f}")

    # Log to wandb
    wandb.log({
        "epoch": epoch + 1,
        f"loss_{fold}": avg_loss
    })
