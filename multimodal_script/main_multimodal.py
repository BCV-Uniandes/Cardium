import torch.nn as nn
import torch
import torch.optim as optim
import wandb
import datetime
import os
import pathlib
import sys
import numpy as np

CARDIUM_path = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(CARDIUM_path))

from multimodal_script.multimodal_models.get_multimodal_model import MultimodalModel
from data.load_multimodal_data import create_multimodal_dataloaders
from utils import *
from data.transformations import transform_train, transform_test
from img_script.img_models.get_img_model import ImageModel
from tabular_script.tab_models.get_tab_model import TabularModel
from multimodal_script.run_multimodal import MultimodalTrainer
from data.dataloader import CardiumDataset

# Parse the arguments
args = get_main_parser()

def main(args):
    """
    Main function to run K-Fold cross-validation training, validation, and testing
    for the multimodal model (image + tabular).

    Args:
        args (Namespace): Parsed arguments containing all configurations.
    """
    
    # Initialize experiment in Weights & Biases
    exp_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    wandb.init(project="MultiModal", 
            entity="spared_v2", 
            name=exp_name)

    wandb.log({"args": vars(args)})

    ######################### TRAIN MODEL ###########################################
    folds = args.folds
    test_metrics = {"F1 Score": [],
                    "Accuracy": [], 
                    "Precision": [],
                    "Recall": []}

    for fold in range(folds):
        print(f"\n=== Fold {fold + 1}/{folds} ===\n")

        # Determine dataset paths
        dataset_path = f"{args.image_folder_path}/fold_{fold+1}"
        json_path = args.json_path

        # Set random seed for reproducibility
        set_seed(42)

        # Create dataloaders
        train_loader, test_loader = create_multimodal_dataloaders(
            dataset_dir=dataset_path,
            json_root=json_path,
            dataset_class=CardiumDataset,
            transform_train=transform_train,
            transform_test=transform_test,
            batch_size=args.batch_size,
            args=args
        )

        print("Dataloaders have been successfully created")

        # --- Build models ---
        # Image model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        img_model = ImageModel(args).build_model().to(device)
        if args.img_checkpoint is not None:
            print("Loading image model pretrained weights...")
            image_checkpoint = os.path.join(args.img_checkpoint, f"fold{fold}_best_model.pth")
            checkpoint = torch.load(image_checkpoint, map_location=device, weights_only=True)
            img_model.load_state_dict(checkpoint, strict=False)

        # Adjust projection layers to Identity
        if args.img_model == "medvit":
            img_model.proj_head = nn.Identity()
        elif args.img_model in ["resnet18", "resnet50"]:
            img_model.fc = nn.Identity()
        else:
            img_model.head = nn.Identity()

        # Tabular model
        tab_model = TabularModel(args).build_model().to(device)
        if args.tab_checkpoint is not None:
            print("Loading tabular model pretrained weights...")
            tab_checkpoint = os.path.join(args.tab_checkpoint, f"fold{fold}_best_model.pth")
            checkpoint = torch.load(tab_checkpoint, map_location=device, weights_only=True)
            tab_model.load_state_dict(checkpoint, strict=False)
        tab_model.mlp = nn.Identity()

        # Multimodal model
        multimodal_model = MultimodalModel(img_model=img_model, tab_model=tab_model, args=args).build_model().to(device)
        if args.multimodal_checkpoint is not None:
            print("Loading multimodal model pretrained weights...")
            multi_checkpoint = os.path.join(args.multimodal_checkpoint, f"fold{fold}_best_model.pth")
            checkpoint = torch.load(multi_checkpoint, map_location=device, weights_only=True)
            multimodal_model.load_state_dict(checkpoint, strict=False)

        # --- Loss function ---
        loss_weights = torch.tensor(args.loss_factor).to(device)

        print(f"Loss weights: {loss_weights}")
        criterion = nn.BCEWithLogitsLoss(pos_weight=loss_weights).to(device)

        # --- Optimizer ---
        optimizer = optim.AdamW(multimodal_model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)

        BASE_DIR = pathlib.Path(__file__).resolve().parent
        save_path = os.path.join(BASE_DIR, "multimodal_checkpoints", exp_name, f"fold{fold}_best_model.pth")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        trainer = MultimodalTrainer(multimodal_model, 
                                    args, 
                                    criterion, 
                                    optimizer, 
                                    device, 
                                    train_loader, 
                                    test_loader, 
                                    fold, 
                                    save_path)
                
        for epoch in range(args.num_epochs):
            print(f"Epoch [{epoch+1}/{args.num_epochs}]")

            # Training phase
            trainer.train_one_epoch(epoch)

            # Validation phase
            trainer.validate_one_epoch()

        # --- Test phase ---
        print("Loading the best model for test evaluation...")
        multimodal_model.load_state_dict(torch.load(save_path))
        f1_test, accuracy_test, precision_test, recall_test = trainer.test_one_epoch(multimodal_model)

        test_metrics["F1 Score"].append(f1_test)
        test_metrics["Accuracy"].append(accuracy_test)
        test_metrics["Precision"].append(precision_test)
        test_metrics["Recall"].append(recall_test)

    # --- Summary ---
    metrics_summary = {key: {"mean": np.mean(values), "std": np.std(values)} for key, values in test_metrics.items()}
    
    print(f"Final average test metrics")
    for metric, summary in metrics_summary.items():
        print(f"{metric}: Mean = {summary['mean']:.4f}, Std = {summary['std']:.4f}")
        wandb.log({f"Average {metric}": summary["mean"], f"Std {metric}": summary["std"]})
    wandb.finish()

if __name__ == "__main__":
    main(args)
