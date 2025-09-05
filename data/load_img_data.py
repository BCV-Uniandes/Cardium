import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import os 
import numpy as np
import random

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)

def create_dataloaders(dataset_dir, dataset_class, transform_train, transform_test, batch_size, args):
    """
    Creates dataloaders for training and testing with weighted sampling for class imbalance.

    Args:
        train_dir (str): Path to training dataset directory.
        val_dir (str): Path to validation dataset directory.
        test_dir (str): Path to testing dataset directory.
        dataset_class (Dataset): Custom dataset class.
        transform_train (callable): Transformations for training dataset.
        transform_test (callable): Transformations for validation and testing datasets.
        batch_size (int): Batch size for the dataloaders.

    Returns:
        tuple: train_loader, test_loader
    """
    # Training dataset
    train_dir = os.path.join(dataset_dir, "train")
    test_dir = os.path.join(dataset_dir, "test")
    
    dataset_train = dataset_class(root=train_dir,  transform=transform_train)
        
    # Calculate class weights for weighted sampling
    if args.sampling:
        print("Implementing sampling")
        num_negatives = sum(1 for _, label in dataset_train if label == 0) #2789 
        num_positives = sum(1 for _, label in dataset_train if label == 1) #470 
        
        class_sample_counts = [num_negatives, num_positives]
        class_weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)

        # Calculate sample weights
        sample_weights = [class_weights[label] for _, label in dataset_train]

        # Create WeightedRandomSampler
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        # Train DataLoader
        train_loader = DataLoader(dataset_train, batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=8, pin_memory=True)
    else:
        print("No sampling implemented")
        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        num_negatives = np.sum(np.array(dataset_train.labels) == 0)
        num_positives = np.sum(np.array(dataset_train.labels) == 1)
        sample_weights = None

    # Test dataset and DataLoader
    dataset_test = dataset_class(root=test_dir, transform=transform_test)    
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return train_loader, test_loader