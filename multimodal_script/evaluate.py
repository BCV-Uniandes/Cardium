import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import wandb
from collections import defaultdict
import pathlib
import sys

delfos_path = pathlib.Path(__name__).resolve().parent.parent
sys.path.append(str(delfos_path))
from utils import *

def evaluate(
    model, 
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
        args: Namespace of arguments (optional, for future extensions).
        mode: "val" or "test" to control logging behavior.
        save_path: Path to save the best model during validation.
        best_f1: Current best F1 score (used during validation to save best model).
        best_threshold: Threshold corresponding to best F1 score.

    Returns:
        best_threshold: Updated threshold corresponding to best F1 (if applicable).
        best_f1: Updated best F1 score (if applicable).
        f1: Current F1 score at patient level.
        avg_loss: Average batch loss (only returned during validation, else None).
    """

    model.eval()  # Set model to evaluation mode
    total_loss = 0.0

    y_true_patient = {}
    y_score_patient = defaultdict(list)

    # --- 1. Iterate over DataLoader ---
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc=f"{mode}ing", unit="batch"):
            img_data, tab_data, patient_ids = inputs[0].to(device), inputs[1].to(device), inputs[2]
            targets = targets.to(device, dtype=torch.float32)

            # Forward pass
            outputs = model(img_data, tab_data).squeeze(1)

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

    # --- 3. Compute patient-level metrics ---
    f1, accuracy, precision, recall = compute_patient_metrics(
        y_score_patient=y_score_patient, 
        y_true_patient=y_true_patient,
        mode=mode,
        fold=fold
    )

    # --- 4. Save best model during validation ---
    if mode.lower() == "val" and best_f1 is not None and f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), save_path)
        print(f"New best model saved with threshold {threshold:.4f} and F1 score: {best_f1:.4f}")

    # --- 5. Return results ---
    return best_f1, accuracy, precision, recall