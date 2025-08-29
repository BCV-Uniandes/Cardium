import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import os 
import numpy as np
import random
import pathlib
import sys

delfos_path = pathlib.Path(__name__).resolve().parent.parent
sys.path.append(str(delfos_path))
from utils import *

torch.backends.cudnn.deterministic = True

# Fix random seed across workers
def worker_init_fn(worker_id):
    seed = 42 + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

def create_dataloaders(
    dataset_dir: str,
    json_root: str,
    dataset_class,
    transform_train: callable,
    transform_test: callable,
    batch_size: int,
    args,
    multimodal: bool = False
) -> tuple:
    """
    Creates DataLoaders for training, validation, and testing with optional weighted sampling
    to handle class imbalance.

    Args:
        dataset_dir (str): Path to the root dataset directory. Should contain 'train', 'val', and 'test' subdirectories.
        json_root (str): Root directory for JSON files used in multimodal datasets.
        dataset_class (Dataset): Custom dataset class that loads data from the specified directories.
        transform_train (callable): A function/transform to apply to the training data.
        transform_test (callable): A function/transform to apply to the validation and test data.
        batch_size (int): Batch size for the DataLoaders.
        args: Arguments containing various options like 'sampling' (bool) for weighted sampling.
        multimodal (bool, optional): Whether the dataset requires JSON files (default is False).

    Returns:
        tuple: A tuple containing:
            - train_loader (DataLoader): DataLoader for training.
            - test_loader (DataLoader): DataLoader for testing.

    Notes:
        - This function assumes a binary classification problem (labels 0 and 1).
        - The function supports optional weighted sampling during training to address class imbalance.
        - If `multimodal` is set to `True`, it assumes the dataset needs additional JSON files.
    """
    # Set a seed for reproducibility
    torch.manual_seed(42)
    generator = torch.Generator()
    generator.manual_seed(42)

    # Decide if worker_init_fn should be used based on multimodal flag
    worker_init_fn_value = worker_init_fn if multimodal else None

    # Directories for dataset splits
    train_dir = os.path.join(dataset_dir, "train")
    test_dir = os.path.join(dataset_dir, "test")

    # Load the training dataset
    if multimodal:
        with tqdm(total=1, desc="Loading Train Dataset") as pbar:
            dataset_train = dataset_class(root=train_dir, json_root=json_root, transform=transform_train)
            pbar.update(1)
    else:
        dataset_train = dataset_class(root=train_dir, transform=transform_train)

    # Calculate class weights if sampling is enabled
    if args.sampling:
        print("Implementing weighted sampling to handle class imbalance.")

        # Count the number of negative and positive samples
        labels_array = np.array(dataset_train.labels)
        num_negatives = np.sum(labels_array == 0)
        num_positives = np.sum(labels_array == 1)

        # Compute class weights
        class_sample_counts = [num_negatives, num_positives]
        class_weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)

        # Calculate sample weights for each instance
        sample_weights = class_weights[labels_array]  # Vectorized indexing
        sample_weights = torch.tensor(sample_weights, dtype=torch.float)

        # Create the WeightedRandomSampler
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True, generator=generator)

        # Create the DataLoader for training with weighted sampling
        train_loader = DataLoader(
            dataset_train,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,  # No need for shuffle if using sampler
            num_workers=8,
            pin_memory=True,
            worker_init_fn=worker_init_fn_value
        )
    else:
        print("No sampling implemented. Using standard shuffled training.")

        # If no sampling is implemented, use regular shuffling
        train_loader = DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            worker_init_fn=worker_init_fn_value
        )

    # Load the test dataset
    if multimodal:
        with tqdm(total=1, desc="Loading Test Dataset") as pbar:
            dataset_test = dataset_class(root=test_dir, json_root=json_root, transform=transform_test)
            pbar.update(1)
    else:
        dataset_test = dataset_class(root=test_dir, transform=transform_test)

    # Create the DataLoader for testing
    test_loader = DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle for testing
        num_workers=8,
        pin_memory=True,
        worker_init_fn=worker_init_fn_value
    )

    # Return the DataLoaders and class counts
    return train_loader, test_loader