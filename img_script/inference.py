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
from data.load_data_imgs import create_dataloaders
from data.img_dataloader import DelfosDataset
from img_script.get_img_model import ImageModel
from img_script.utils_imgs import *


def main(args):
    folds = args.folds # Number of folds for cross-validation

    # Initialize lists to store metrics for each fold
    metrics_folds ={"precision":[],
                    "recall": [],
                    "f1-score": [],
                    "roc auc": []}

    # Cross-validation
    for fold in range(folds):
        set_seed(42) # Set random seed for reproducibility
        # Set the dataset path for the current fold
        dataset_path = f"{args.data_path}/fold_{fold+1}"
        # Create data loaders for the current fold
        _, _, test_loader, _, _, _ = create_dataloaders(
            dataset_dir = dataset_path,
            json_root="",
            dataset_class=DelfosDataset,
            transform_train=transform_train,
            transform_test=transform_test,
            batch_size=args.batch_size,
            fold=fold,
            args=args
        )
        
        # Load multimodal model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        #Get image model
        img_model = ImageModel(args)
        img_model = img_model.build_model()
        img_model = img_model.to(device)

        exp_name = args.exp_name
        img_checkpoint = os.path.join(args.img_checkpoint, exp_name, f"fold{fold}_best_model.pth")
        img_checkpoint = torch.load(img_checkpoint, map_location = torch.device("cuda"), weights_only=True)
        img_model.load_state_dict(img_checkpoint, strict=False)
        img_model = img_model.to(device)

        #mean_threshold = 0.5
        precision_test, recall_test, _, f1_test, roc_auc_test, _, _, _ = evaluate_threshold_img(img_model, test_loader, device)
        #print metrics for folds
        print(f"Test Metrics_{fold}:")
        print(f"Precision: {precision_test:.4f}")
        print(f"Recall: {recall_test:.4f}")
        print(f"F1-Score: {f1_test:.4f}")
        print(f"ROC AUC: {roc_auc_test:.4f}")
        
        metrics_folds["precision"].append(precision_test)
        metrics_folds["recall"].append(recall_test)
        metrics_folds["f1-score"].append(f1_test)
        metrics_folds["roc auc"].append(roc_auc_test)
        
    # Compute mean and standard deviation for each metric
    metrics_summary = {}
    for key, values in metrics_folds.items():
        mean_value = np.mean(values)
        std_value = np.std(values)
        metrics_summary[key] = {"mean": mean_value, "std": std_value}

    # Print the results
    for metric, summary in metrics_summary.items():
        print(f"{metric}: Mean = {summary['mean']:.4f}, Std = {summary['std']:.4f}")

if __name__ == "__main__":
    # Parse the arguments
    args = get_main_parser()
    # Use the provided experiment name
    exp_name = args.exp_name
    # Start the main function
    main(args)