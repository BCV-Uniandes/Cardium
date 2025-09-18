import os
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import wandb
from utils import *

class TabularTrainer:
    def __init__(self, model, args, criterion, optimizer, scheduler, device,
                 train_loader, val_loader, train_dataset,
                positives, fold_idx, save_path, hn_epochs=20):
        """
        Training and validation loop for tabular classification models 
        with support for Hard Negative Mining (HNM).

        Args:
            model (torch.nn.Module): TabEncoder model.
            criterion (torch.nn.Module): Loss function (BCEWithLogitsLoss).
            optimizer (torch.optim.Optimizer): Optimizer for training (AdamW).
            scheduler (torch.optim.lr_scheduler): Learning rate scheduler (StepLR).
            device (torch.device): Device to run training on ("cpu" or "cuda").
            train_loader (DataLoader): DataLoader for the training set.
            val_loader (DataLoader): DataLoader for the validation set.
            train_dataset (torch.utils.data.Dataset): Training dataset (needed for HNM resampling).
            positives (int): Number of positive samples in the training set.
                            Used to scale false negatives during HNM.
            fold_idx (int): Fold index for cross-validation (used in filenames).
            save_path (str): Path to save best model.
            hn_epochs (int, optional): Frequency (in epochs) at which to apply HNM. Default: 20
        """
        self.model = model
        self.args = args
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_dataset = train_dataset

        self.positives = positives
        self.fold_idx = fold_idx
        self.save_path = save_path
        self.hn_epochs = hn_epochs

        self.best_f1_val = 0
        self.best_threshold = 0.5

    def train_one_epoch(self, epoch):
        """
        Run one training epoch over the training set.

        Returns:
            float: Average training loss across all batches.
        """
        self.model.train()
        running_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.num_epochs}", unit="batch")

        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs).squeeze()
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix({
            "Batch Loss": f"{loss.item():.4f}",
            "Avg Loss": f"{running_loss / (batch_idx + 1):.4f}"
            })

        avg_loss = running_loss / len(self.train_loader)
        wandb.log({
                "epoch": epoch + 1,
                f"loss_{self.fold_idx}": avg_loss,
            })

    def validate_one_epoch(self):
        """
        Run one validation epoch and compute F1 score.

        Returns:
            tuple:
                avg_loss (float): Average validation loss.
                f1_val (float): Validation F1 score at fixed threshold (0.5).
                val_probs (np.ndarray): Predicted probabilities for validation set.
        """
        self.model.eval()
        running_loss = 0.0
        val_probs, val_true = [], []
        # --- 1. Iterate over DataLoader ---
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc=f"Validating", unit="batch"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs).squeeze()
                loss = self.criterion(outputs, targets)
                running_loss += loss.item()
                outputs_prob = torch.sigmoid(outputs)
                val_probs.extend(outputs_prob.cpu().numpy())
                val_true.extend(targets.cpu().numpy())

        # --- 2. Average loss across batches ---
        avg_loss = running_loss / len(self.val_loader)
        wandb.log({"val_loss": avg_loss})

        # --- 3. Convert lists to arrays for metric computation ---
        val_probs = np.array(val_probs)
        val_true = np.array(val_true)

        # --- 4. Compute patient-level metrics ---
        f1, _, _, _ = compute_patient_metrics(
            y_score=val_probs, 
            y_true=val_true,
            mode="val",
            fold=self.fold_idx
        )

        # --- 7. Save best model during validation ---
        if f1 > self.best_f1_val:
            self.best_f1_val = f1
            torch.save(self.model.state_dict(), self.save_path)
            print(f"New best model saved with an F1 score: {self.best_f1_val:.4f}")

        print(f"Best F1 so far: {self.best_f1_val}")

    def test_one_epoch(self, best_model):
        """
        Run one validation epoch and compute F1 score.

        Returns:
            tuple:
                avg_loss (float): Average validation loss.
                f1_val (float): Validation F1 score at fixed threshold (0.5).
                val_probs (np.ndarray): Predicted probabilities for validation set.
        """
        best_model.eval()
        running_loss = 0.0
        test_probs, test_true = [], []
        # --- 1. Iterate over DataLoader ---
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc=f"Testing", unit="batch"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = best_model(inputs).squeeze()
                loss = self.criterion(outputs, targets)
                running_loss += loss.item()
                outputs_prob = torch.sigmoid(outputs)
                test_probs.extend(outputs_prob.cpu().numpy())
                test_true.extend(targets.cpu().numpy())

        # --- 2. Convert lists to arrays for metric computation ---
        test_probs = np.array(test_probs)
        test_true = np.array(test_true)

        # --- 4. Compute patient-level metrics ---
        f1, accuracy, precision, recall = compute_patient_metrics(
            y_score=test_probs, 
            y_true=test_true,
            mode="test",
            fold=self.fold_idx
        )

        return f1, accuracy, precision, recall 

    def hard_negative_mining(self):
        """
        Apply Hard Negative Mining (HNM) to rebalance the training loader.

        """
        self.model.eval()
        all_preds, y_train_true = [], []
        temp_loader = DataLoader(
            self.train_dataset, batch_size=self.args.batch_size, shuffle=False,
            num_workers=0, worker_init_fn=lambda x: worker_init_fn(x, self.args.seed)
        )

        with torch.no_grad():
            for inputs, targets in temp_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs).squeeze()
                outputs_prob = torch.sigmoid(outputs)
                all_preds.append(outputs_prob.cpu().numpy())
                y_train_true.append(targets.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        y_train_true = np.concatenate(y_train_true)

        false_negatives_mask = (y_train_true == 1) & (all_preds <= self.best_threshold)
        fn_indices = np.where(false_negatives_mask)[0]

        new_sample_weights = torch.ones(len(self.train_dataset))
        if len(fn_indices) > 0:
            fn_weight = 1 + (len(fn_indices) / self.positives)
            new_sample_weights[fn_indices] = fn_weight

        sampler = WeightedRandomSampler(
            new_sample_weights, len(new_sample_weights),
            replacement=True, generator=torch.Generator().manual_seed(self.args.seed)
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=0,
            worker_init_fn=lambda x: worker_init_fn(x, self.args.seed)
        )
        print(f"Hard Negative Mining applied - False Negatives: {len(fn_indices)}")

    def train(self, epochs):
        """
        Full training loop across multiple epochs.

        Args:
            epochs (int): Number of training epochs.
        """
        for epoch in range(epochs):
            print(f"Epoch [{epoch+1}/{epochs}]")
            self.train_one_epoch(epoch=epoch)
            self.validate_one_epoch()
            self.scheduler.step()

            if (epoch + 1) % self.hn_epochs == 0 and epoch != 0:
                self.hard_negative_mining()




