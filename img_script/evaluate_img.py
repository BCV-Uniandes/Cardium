import torch
from tqdm import tqdm
import wandb
from collections import defaultdict
import pathlib
import sys
import numpy as np

delfos_path = pathlib.Path(__name__).resolve().parent.parent
sys.path.append(str(delfos_path))
from utils import *

def evaluate(model, 
             loader, 
             criterion, 
             device, 
             fold, 
             mode: str = "val", 
             save_path: str = None, 
             best_f1: float = None, 
             threshold: float = None
):
    """
    Evaluate the model at the patient level and compute metrics.

    Args:
        model: Multimodal model.
        loader: DataLoader providing batches of (image, tabular, patient_ids) and targets.
        criterion: Loss function.
        device: Device for computation ('cpu' or 'cuda').
        fold: Identifier of the current fold (for logging).
        mode: "val" or "test" to control logging behavior.
        save_path: Path to save the best model during validation.
        best_f1: Current best F1 score (used during validation to save best model).
        threshold: Threshold to compute metrics

    Returns:
        best_f1: Updated best F1 score (if applicable).
        f1: Current F1 score at patient level.
        accuracu: Current accuracy at patient level
        precision: Current precision at patient level
        recall: Current recall at patient level
    """

    # Evaluate the model on the validation or test set
    model.eval()
    # Initialize variables to track metrics
    total_loss = 0.0
    y_true_patient = {}
    y_score_patient = defaultdict(list)
    
    # --- 1. Iterate over DataLoader ---
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc=f"{mode}ing", unit="batch"):
            # Move inputs and targets to the appropriate device
            inputs, patient_ids = inputs[0].to(device), inputs[1]
            targets = targets.unsqueeze(1).to(device)
            targets = targets.to(torch.float32)
            
            # Forward pass
            outputs = model(inputs)
    
            # Compute loss
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # Convert logits to probabilities
            probabilities = torch.sigmoid(outputs).cpu().numpy()
            targets_np = targets.cpu().numpy()

            # Aggregate scores and true labels by patient ID
            for pid, prob, target in zip(patient_ids, probabilities, targets_np):
                y_score_patient[pid].append(prob)
                y_true_patient[pid] = target  # True label is same for all occurrences of patient

    # --- 2. Average loss across batches ---
    avg_loss = total_loss / len(loader)
    print(f"{mode} Loss: {avg_loss:.4f}")
    wandb.log({f"{mode.lower()}_loss": avg_loss})

    # --- 3. Average scores per patient ---
    y_score_patient_avg = {pid: np.mean(scores) for pid, scores in y_score_patient.items()}
    y_true_patient = {pid: label for pid, label in y_true_patient.items()}

    # --- 4. Convert dictionaries to arrays for metric computation ---
    y_true = np.array(list(y_true_patient.values()))
    y_score = np.array(list(y_score_patient_avg.values()))

    # --- 5. Compute patient-level metrics ---
    f1, accuracy, precision, recall = compute_patient_metrics(
        y_score=y_score, 
        y_true=y_true,
        mode=mode,
        fold=fold
    )

    # --- 6. Save best model during validation ---
    if mode.lower() == "val" and best_f1 is not None and f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), save_path)
        print(f"New best model saved with an F1 score: {best_f1:.4f}")

    # --- 7. Return results ---
    return best_f1, f1, accuracy, precision, recall