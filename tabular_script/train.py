import os
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score
import wandb
from tabular_script.tab_utils import worker_init_fn

class TabularTrainer:
    def __init__(self, model, criterion, optimizer, scheduler, device,
                 train_loader, val_loader, train_dataset,
                 batch_size, seed, positives,
                 weights_dir, fold_idx, hn_epochs=20):
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
            batch_size (int): Batch size used for loaders.
            seed (int): Random seed for reproducibility.
            positives (int): Number of positive samples in the training set.
                            Used to scale false negatives during HNM.
            weights_dir (str or Path): Directory where best model weights will be saved.
            fold_idx (int): Fold index for cross-validation (used in filenames).
            hn_epochs (int, optional): Frequency (in epochs) at which to apply HNM. Default: 20
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_dataset = train_dataset

        self.batch_size = batch_size
        self.seed = seed
        self.positives = positives
        self.weights_dir = weights_dir
        self.fold_idx = fold_idx
        self.hn_epochs = hn_epochs

        os.makedirs(weights_dir, exist_ok=True)
        self.best_f1_val = 0
        self.best_threshold = 0.5
        self.fold_val_probs = []

    def train_one_epoch(self):
        """
        Run one training epoch over the training set.

        Returns:
            float: Average training loss across all batches.
        """
        self.model.train()
        total_loss = 0
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs).squeeze()
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

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
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs).squeeze()
                loss = self.criterion(outputs, targets)
                running_loss += loss.item()
                outputs_prob = torch.sigmoid(outputs)
                val_probs.extend(outputs_prob.cpu().numpy())
                val_true.extend(targets.cpu().numpy())

        avg_loss = running_loss / len(self.val_loader)
        val_probs = np.array(val_probs)
        val_true = np.array(val_true)

        y_pred_bin = (val_probs > self.best_threshold).astype(float)
        f1_val = f1_score(val_true, y_pred_bin, pos_label=1)

        return avg_loss, f1_val, val_probs

    def hard_negative_mining(self):
        """
        Apply Hard Negative Mining (HNM) to rebalance the training loader.

        """
        self.model.eval()
        all_preds, y_train_true = [], []
        temp_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=0, worker_init_fn=lambda x: worker_init_fn(x, self.seed)
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
            replacement=True, generator=torch.Generator().manual_seed(self.seed)
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=0,
            worker_init_fn=lambda x: worker_init_fn(x, self.seed)
        )
        print(f"Hard Negative Mining applied - False Negatives: {len(fn_indices)}")

    def train(self, epochs):
        """
        Full training loop across multiple epochs.

        Args:
            epochs (int): Number of training epochs.

        Returns:
            tuple:
                best_f1_val (float): Best validation F1 score achieved.
                fold_val_probs (np.ndarray): Probabilities for validation set at best epoch.
        """
        for epoch in range(epochs):
            train_loss = self.train_one_epoch()
            val_loss, f1_val, val_probs = self.validate_one_epoch()
            self.scheduler.step()

            print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val F1: {f1_val:.4f}")

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_f1": f1_val,
            })

            if f1_val > self.best_f1_val:
                self.best_f1_val = f1_val
                self.fold_val_probs = val_probs.copy()
                model_path = os.path.join(self.weights_dir, f'best_model_fold_{self.fold_idx+1}.pth')
                torch.save(self.model.state_dict(), model_path)
                print(f"Model saved to {model_path} with F1-Score: {self.best_f1_val:.4f}")

            if (epoch + 1) % self.hn_epochs == 0 and epoch != 0:
                self.hard_negative_mining()

        return self.best_f1_val, self.fold_val_probs



