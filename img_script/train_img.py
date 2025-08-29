import torch
from tqdm import tqdm
import wandb

def train_one_epoch(model, train_loader, criterion, optimizer, epoch, device, fold, args):
    # Train the model for one epoch
    model.train()
    # Track the running loss
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}", unit="batch")
    # Loop through the data
    for inputs, targets in progress_bar:
        # Move inputs and targets to the appropriate device
        inputs, targets = inputs[0].to(device), targets.unsqueeze(1).to(device)

        # Forward + backward + optimize
        optimizer.zero_grad()
        outputs = model(inputs)
        targets = targets.to(torch.float32)
        # Compute the loss
        loss = criterion(outputs, targets)
        # Backpropagation
        loss.backward()
        # Update the model parameters
        optimizer.step()
        # Update the running loss
        running_loss += loss.item()
        progress_bar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})

    # Calculate the average loss for the epoch
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{args.num_epochs}], Average Loss: {avg_loss:.4f}")
    # Log the average loss to wandb
    wandb.log({"epoch": epoch+1, f"loss_{fold}": avg_loss})
