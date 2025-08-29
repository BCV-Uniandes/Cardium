import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import wandb
from collections import defaultdict
import pathlib
import sys
import numpy as np

delfos_path = pathlib.Path(__name__).resolve().parent.parent
sys.path.append(str(delfos_path))
from get_metrics import *

def evaluate(model, loader, criterion, device, fold, args, mode="val", save_path=None, best_f1=None):
    # Evaluate the model on the validation or test set
    model.eval()
    # Initialize variables to track metrics
    total_loss = 0.0
    y_true_patient = {}
    y_score_patient = defaultdict(list)
    # Loop through the data
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc=f"{mode}ing", unit="batch"):
            # Move inputs and targets to the appropriate device
            inputs, patient_ids = inputs[0].to(device), inputs[1]
            targets = targets.unsqueeze(1).to(device)
            # Forward pass
            outputs = model(inputs)
            targets = targets.to(torch.float32)
            # Compute and track the loss
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # Sigmoid to convert logits to probabilities
            probabilities = torch.sigmoid(outputs).cpu().numpy()
            targets = targets.cpu().numpy()

            # Group predictions and true labels by patient_id
            for patient_id, prob, target in zip(patient_ids, probabilities, targets):
                if patient_id not in y_score_patient:
                    y_score_patient[patient_id] = np.array([prob])  # Initialize as array
                else:
                    y_score_patient[patient_id] = np.append(y_score_patient[patient_id], prob)  # Append to array

                y_true_patient[patient_id] = target  # Store the true label

    # Compute metrics
    avg_loss = total_loss / len(loader)
    print(f"{mode} Loss: {avg_loss:.4f}")
    # Log metrics to wandb
    wandb.log({f"{mode.lower()}_loss": avg_loss})
    # Compute patient-level metrics
    f1, _ = compute_patient_metrics(y_score_patient=y_score_patient, 
                                 y_true_patient=y_true_patient,
                                 mode=mode,
                                 fold=fold)
    # Save the best model if applicable
    if mode == "val" and best_f1 is not None and f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), save_path)
        print(f"New best model saved with F1 score: {best_f1:.4f}")

    return best_f1, f1, avg_loss if mode == "val" else None