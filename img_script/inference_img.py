import torch.nn as nn
import torch
import torch.optim as optim
from tqdm import tqdm
import wandb
from datetime import datetime
import os
import pathlib
import sys
import numpy as np
import matplotlib.pyplot as plt

delfos_path = pathlib.Path(__name__).resolve().parent.parent
sys.path.append(str(delfos_path))
from data.transformations import transform_train, transform_test
from data.load_img_data import create_dataloaders
from data.img_dataloader import DelfosDataset
from data.dataloader import CardiumDataset
from img_script.img_models.get_img_model import ImageModel
from utils import *

def main(args):
    """
    Main function to run K-Fold cross-validation inference only
    for the multimodal model (image + tabular).

    Args:
        args (Namespace): Parsed arguments containing all configurations.
    """
    
    # Prepare metrics storage
    metrics_folds ={"precision":[],
                    "recall": [],
                    "f1-score": [],
                    "accuracy": []}

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Cross-validation
    for fold in range(args.folds):
        print(f"Processing fold {fold}")
        set_seed(42) 
        
        # Set the dataset path for the current fold
        dataset_path = f"{args.image_folder_path}/fold_{fold+1}" # Without trimester separation

        # Create data loaders for the current fold
        _, test_loader = create_dataloaders(
            dataset_dir = dataset_path,
            dataset_class=CardiumDataset,
            transform_train=transform_train,
            transform_test=transform_test,
            batch_size=args.batch_size,
            args=args
        )
        
        # Get image model
        img_model = ImageModel(args)
        img_model = img_model.build_model()
        img_model = img_model.to(device)

        img_checkpoint = os.path.join(args.img_checkpoint, f"fold{fold}_best_model.pth")
        img_checkpoint = torch.load(img_checkpoint, map_location = torch.device("cuda"), weights_only=True)
        img_model.load_state_dict(img_checkpoint, strict=False)
        img_model = img_model.to(device)

        # Inference
        y_true, y_score = inference_image_model(img_model, test_loader, device)
        f1, accuracy, precision, recall = compute_patient_metrics(y_score=y_score, y_true=y_true, 
                                                                  mode="test", fold=fold, log_wandb=False)
        
        # Store metrics for this fold
        metrics_folds["precision"].append(precision)
        metrics_folds["recall"].append(recall)
        metrics_folds["f1-score"].append(f1)
        metrics_folds["accuracy"].append(accuracy)

    # Compute and log mean and std of metrics
    metrics_summary = {key: {"mean": np.mean(values), "std": np.std(values)} for key, values in metrics_folds.items()}
    
    print(f"Final average test metrics")
    for metric, summary in metrics_summary.items():
        print(f"{metric}: Mean = {summary['mean']:.4f}, Std = {summary['std']:.4f}")

if __name__ == "__main__":
    # Assuming get_main_parser() is defined elsewhere
    args = get_main_parser()
    main(args)