import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from tabular_script.tab_utils import *

class TabularDataLoaders:
    def __init__(self, fold_idx, base_dir, seed, sampling, batch_size):
        """
        Utility for loading tabular cross-validation folds and creating 
        training and validation dataloaders.

        Args:
            fold_idx (int):
                Index of the current fold to load (used for cross-validation).
            base_dir (str or Path):
                Base directory where preprocessed fold data is stored.
            seed (int):
                Random seed for reproducibility.
            sampling (float):
                Sampling factor for balancing classes. 
                - If 0, no weighted sampling is applied.
                - Otherwise, the positive class is weighted by this factor.
            batch_size (int):
                Number of samples per batch in both train and validation loaders.
        """
        self.fold_idx = fold_idx
        self.base_dir = base_dir
        self.seed = seed
        self.sampling = sampling
        self.batch_size = batch_size

        self.X_train, self.y_train, self.X_val, self.y_val = load_fold(
            self.fold_idx, self.base_dir, self.seed
        )

    def _create_sampler(self):
        """
        Create a WeightedRandomSampler for the training set 
        to handle class imbalance.

        Returns:
            torch.utils.data.WeightedRandomSampler or None:
                - Weighted sampler if `self.sampling > 0`.
                - None if no sampling is applied.
        """
        
        num_neg = (self.y_train == 0).sum()
        num_pos = (self.y_train == 1).sum()
        class_sample_counts = [num_neg, num_pos * self.sampling]

        class_weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
        labels_array = np.array(self.y_train)
        sample_weights = class_weights[labels_array]
        sample_weights = torch.tensor(sample_weights, dtype=torch.float)
        if self.sampling == 0:
            return None
        else:
            return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    def get_loaders(self):
        """
        Create and return DataLoaders for training and validation sets.

        Returns:
            tuple:
                - X_train (np.ndarray): Training features.
                - train_loader (DataLoader): PyTorch DataLoader for training set.
                - val_loader (DataLoader): PyTorch DataLoader for validation set.
                - pos (int): Number of positive samples in training set.
                - neg (int): Number of negative samples in training set.
                - train_dataset (TensorDataset): Training dataset (features and labels).
                - val_dataset (TensorDataset): Validation dataset (features and labels).
                - y_val (np.ndarray): Validation labels (ground truth).
        """
        sampler = self._create_sampler()
        neg = int((self.y_train == 0).sum())
        pos = int((self.y_train == 1).sum())

        train_dataset = TensorDataset(
            torch.tensor(self.X_train, dtype=torch.float32),
            torch.tensor(self.y_train, dtype=torch.float32)
        )
        val_dataset = TensorDataset(
            torch.tensor(self.X_val, dtype=torch.float32),
            torch.tensor(self.y_val, dtype=torch.float32)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False, 
            num_workers=0,
            worker_init_fn=lambda x: worker_init_fn(x, self.seed),
            sampler=sampler
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            worker_init_fn=lambda x: worker_init_fn(x, self.seed)
        )

        return self.X_train, train_loader, val_loader, pos, neg, train_dataset, val_dataset, self.y_val
