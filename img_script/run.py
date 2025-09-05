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


class ImageTrainer:
    def __init__(self, model, args, criterion, optimizer, device,
                 train_loader, test_loader, fold, save_path):
        """
        Training and validation loop for multimodal model.

        Args:
            model (torch.nn.Module): TabEncoder model.
            criterion (torch.nn.Module): Loss function (BCEWithLogitsLoss).
            optimizer (torch.optim.Optimizer): Optimizer for training (AdamW).
            device (torch.device): Device to run training on ("cpu" or "cuda").
            train_loader (DataLoader): DataLoader for the training set.
            val_loader (DataLoader): DataLoader for the validation set.
            train_dataset (torch.utils.data.Dataset): Training dataset (needed for HNM resampling).
            fold (int): Fold index for cross-validation (used in filenames).
            save_path (str): Path to save best model.
        """
        self.model = model
        self.args = args
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.fold = fold
        self.save_path = save_path

        self.best_f1 = 0

    def train_one_epoch(self, epoch):
        """
        Train the model for one epoch.

        """
        
        # Train the model for one epoch
        self.model.train()
        
        # Track the running loss
        running_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.num_epochs}", unit="batch")
        
        # --- 1. Iterate over DataLoader ---
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            # Move inputs and targets to the appropriate device
            inputs, targets = inputs[0].to(self.device), targets.unsqueeze(1).to(self.device)
            targets = targets.to(torch.float32)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Compute the loss
            loss = self.criterion(outputs, targets)
            
            # Backpropagation
            loss.backward()
            
            # Update the model parameters
            self.optimizer.step()
            
            # Update the running loss
            running_loss += loss.item()
            progress_bar.set_postfix({
                "Batch Loss": f"{loss.item():.4f}",
                "Avg Loss": f"{running_loss / (batch_idx + 1):.4f}"
                })

        # Compute average loss for the epoch
        avg_loss = running_loss / len(self.train_loader)
        print(f"Epoch [{epoch+1}/{self.args.num_epochs}], Average Loss: {avg_loss:.4f}")
        
        # Log to wandb
        wandb.log({
            "epoch": epoch+1, 
            f"loss_{self.fold}": avg_loss
        })

    def validate_one_epoch(self):
        """
        Validate the model at the patient level and compute metrics.

        """

        # Evaluate the model on the validation or test set
        self.model.eval()
        # Initialize variables to track metrics
        total_loss = 0.0
        y_true_patient = {}
        y_score_patient = defaultdict(list)
        
        # --- 1. Iterate over DataLoader ---
        with torch.no_grad():
            for inputs, targets in tqdm(self.test_loader, desc=f"Validating", unit="batch"):
                # Move inputs and targets to the appropriate device
                inputs, patient_ids = inputs[0].to(self.device), inputs[1]
                targets = targets.unsqueeze(1).to(self.device)
                targets = targets.to(torch.float32)
                
                # Forward pass
                outputs = self.model(inputs)
        
                # Compute loss
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

                # Convert logits to probabilities
                probabilities = torch.sigmoid(outputs).cpu().numpy()
                targets_np = targets.cpu().numpy()

                # Aggregate scores and true labels by patient ID
                for pid, prob, target in zip(patient_ids, probabilities, targets_np):
                    y_score_patient[pid].append(prob)
                    y_true_patient[pid] = target  # True label is same for all occurrences of patient

        # --- 2. Average loss across batches ---
        avg_loss = total_loss / len(self.test_loader)
        print(f"val Loss: {avg_loss:.4f}")
        wandb.log({"val_loss": avg_loss})

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
            mode="val",
            fold=self.fold
        )

        # --- 6. Save best model during validation ---
        if f1 > self.best_f1:
            self.best_f1 = f1
            torch.save(self.model.state_dict(), self.save_path)
            print(f"New best model saved with an F1 score: {self.best_f1:.4f}")

    def test_one_epoch(self, model):
        """
        Evaluate the model at the patient level and compute metrics.

        """

        # Evaluate the model on the validation or test set
        model.eval()
        # Initialize variables to track metrics
        total_loss = 0.0
        y_true_patient = {}
        y_score_patient = defaultdict(list)
        
        # --- 1. Iterate over DataLoader ---
        with torch.no_grad():
            for inputs, targets in tqdm(self.test_loader, desc=f"Testing", unit="batch"):
                # Move inputs and targets to the appropriate device
                inputs, patient_ids = inputs[0].to(self.device), inputs[1]
                targets = targets.unsqueeze(1).to(self.device)
                targets = targets.to(torch.float32)
                
                # Forward pass
                outputs = model(inputs)
        
                # Compute loss
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

                # Convert logits to probabilities
                probabilities = torch.sigmoid(outputs).cpu().numpy()
                targets_np = targets.cpu().numpy()

                # Aggregate scores and true labels by patient ID
                for pid, prob, target in zip(patient_ids, probabilities, targets_np):
                    y_score_patient[pid].append(prob)
                    y_true_patient[pid] = target  # True label is same for all occurrences of patient

        # --- 2. Average scores per patient ---
        y_score_patient_avg = {pid: np.mean(scores) for pid, scores in y_score_patient.items()}
        y_true_patient = {pid: label for pid, label in y_true_patient.items()}

        # --- 3. Convert dictionaries to arrays for metric computation ---
        y_true = np.array(list(y_true_patient.values()))
        y_score = np.array(list(y_score_patient_avg.values()))

        # --- 4. Compute patient-level metrics ---
        f1, accuracy, precision, recall = compute_patient_metrics(
            y_score=y_score, 
            y_true=y_true,
            mode="test",
            fold=self.fold
        )

        return f1, accuracy, precision, recall