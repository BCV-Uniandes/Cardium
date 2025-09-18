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

#set_seed(42)
#generator = torch.Generator()
#generator.manual_seed(42)  
torch.backends.cudnn.deterministic = True

# Fix random seed across workers
def worker_init_fn(worker_id):
    seed = 42 + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def create_multimodal_dataloaders(dataset_dir, json_root, dataset_class, transform_train, transform_test, batch_size, args):
    """
    Creates dataloaders for training, validation, and testing with weighted sampling for class imbalance.

    Args:
        train_dir (str): Path to training dataset directory.
        val_dir (str): Path to validation dataset directory.
        test_dir (str): Path to testing dataset directory.
        dataset_class (Dataset): Custom dataset class.
        transform_train (callable): Transformations for training dataset.
        transform_test (callable): Transformations for validation and testing datasets.
        batch_size (int): Batch size for the dataloaders.
 
    Returns:
        tuple: train_loader, val_loader, test_loader
    """
    set_seed(42)
    generator = torch.Generator()
    generator.manual_seed(42)  

    # Train and test directories
    train_dir = os.path.join(dataset_dir, "train")
    test_dir = os.path.join(dataset_dir, "test")
    
    # Train dataset
    dataset_train = dataset_class(root=train_dir, transform=transform_train, json_root=json_root, multimodal=True)

    # Calculate class weights for weighted sampling and define dataloader
    if args.sampling:
        print("Implementing sampling")
        # Count the number of negatives and positives with tqdm
        num_negatives = np.sum(np.array(dataset_train.labels) == 0)
        num_positives = np.sum(np.array(dataset_train.labels) == 1)

        
        class_sample_counts = [num_negatives, num_positives]
        class_weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)

        # Calculate sample weights
        labels_array = np.array(dataset_train.labels)  # Convert list to NumPy array
        sample_weights = class_weights[labels_array]  # Vectorized indexing

        # Convert to PyTorch tensor (optional)
        sample_weights = torch.tensor(sample_weights, dtype=torch.float)
        
        # Create WeightedRandomSampler
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True, generator=generator)

        # Train DataLoader
        train_loader = DataLoader(dataset_train, batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    else:
        print("No sampling implemented")
        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
        num_negatives = np.sum(np.array(dataset_train.labels) == 0)
        num_positives = np.sum(np.array(dataset_train.labels) == 1)
        sample_weights = None
    
    # Test dataset and dataLoader
    dataset_test = dataset_class(root=test_dir, json_root=json_root, transform=transform_train, multimodal=True)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)

    return train_loader, test_loader
