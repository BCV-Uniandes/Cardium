import torch
import os
import pathlib
import sys
import numpy as np

CARDIUM_path = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(CARDIUM_path))

from data.transformations import transform_train, transform_test
from data.dataloader import CardiumDataset
from img_script.img_models.get_img_model import ImageModel
from tabular_script.tab_models.get_tab_model import TabularModel
from multimodal_script.multimodal_models.get_multimodal_model import MultimodalModel
from data.load_multimodal_data import create_multimodal_dataloaders
from utils import *


def load_checkpoint(model, checkpoint_path, strict=False):
    """Helper function to load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cuda"), weights_only=True)
    model.load_state_dict(checkpoint, strict=strict)
    print(f"Model weights loaded from {checkpoint_path}")
    return model

def main(args):
    """
    Main function to run K-Fold cross-validation inference only
    for the multimodal model (image + tabular).

    Args:
        args (Namespace): Parsed arguments containing all configurations.
    """
    
    # Prepare metrics storage
    metrics_folds = {"precision": [], 
                     "recall": [], 
                     "f1-score": [], 
                     "accuracy": []}
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Cross-validation
    for fold in range(args.folds):
        print(f"Processing fold {fold}")
        set_seed(42) 
        if args.trimester == None:
            dataset_path = f"{args.image_folder_path}/fold_{fold+1}"
        else:
            dataset_path = f"{args.image_folder_path}/{args.trimester}_trimester/fold_{fold+1}"
        
        json_path = args.json_path

         # Create dataloaders
        _, test_loader = create_multimodal_dataloaders(
            dataset_dir=dataset_path,
            json_root=json_path,
            dataset_class=CardiumDataset,
            transform_train=transform_train,
            transform_test=transform_test,
            batch_size=args.batch_size,
            args=args
        )

        # Load image model
        img_model = ImageModel(args).build_model().to(device)
        if args.img_checkpoint:
            img_checkpoint_path = os.path.join(args.img_checkpoint, f"fold{fold}_best_model.pth")
            img_model = load_checkpoint(img_model, img_checkpoint_path, strict=False)

        if args.img_model == "medvit":
            img_model.proj_head = torch.nn.Identity()
        elif args.img_model in ["resnet18", "resnet50"]:
            img_model.fc = torch.nn.Identity()
        else:
            img_model.head = torch.nn.Identity()

        # Load tabular model
        tab_model = TabularModel(args).build_model().to(device)
        if args.tab_checkpoint:
            tab_checkpoint_path = os.path.join(args.tab_checkpoint, f"fold{fold}_best_model.pth")
            tab_model = load_checkpoint(tab_model, tab_checkpoint_path, strict=False)
        
        tab_model.mlp = torch.nn.Identity()

        # Load multimodal model
        multimodal_model = MultimodalModel(img_model=img_model, tab_model=tab_model, args=args).build_model().to(device)
        multimodal_checkpoint_path = os.path.join(args.multimodal_checkpoint, f"fold{fold}_best_model.pth")
        multimodal_model = load_checkpoint(multimodal_model, multimodal_checkpoint_path, strict=True)

        # Inference
        y_true, y_score = inference_multimodal(multimodal_model, test_loader, device)
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