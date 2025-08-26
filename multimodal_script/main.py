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

delfos_path = pathlib.Path(__name__).resolve().parent.parent
sys.path.append(str(delfos_path))
from utils import *
from data.transformations import transform_train, transform_test
from data.load_data import create_dataloaders
from data.multimodal_dataloader import DelfosDataset
from img_script.get_img_model import ImageModel
from tabular_script.get_tab_model import TabularModel
from multimodal_script.get_multimodal_model import MultimodalModel
from multimodal_script.train_multimodal import train_one_epoch
from multimodal_script.evaluate_multimodal import evaluate
from utils import *

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

    config = wandb.config
    wandb.log({"args": vars(args),
            "model": "multimodal_kfold"})

    folds = args.folds
    best_f1_folds = []

    for fold in range(folds):
        print(f"\n=== Fold {fold + 1}/{folds} ===\n")

        # Determine dataset paths
        if args.trimester is None:
            dataset_path = f"/home/dvegaa/DELFOS/CARDIUM/dataset/delfos_images_kfold/fold_{fold+1}"
        else:
            dataset_path = f"/home/dvegaa/DELFOS/CARDIUM/trimester_analisis/dataset_correct_trimesters/{args.trimester}_trimester/fold_{fold+1}"
        json_root = "/home/dvegaa/DELFOS/CARDIUM/dataset/delfos_clinical_data_woe_wnm_standarized_f_normalized.json"

        # Set random seed for reproducibility
        set_seed(42)

        # Create dataloaders
        train_loader, _, test_loader, num_negatives, num_positives, _ = create_dataloaders(
            dataset_dir=dataset_path,
            json_root=json_root,
            dataset_class=DelfosDataset,
            transform_train=transform_train,
            transform_test=transform_test,
            batch_size=args.batch_size,
            fold=fold,
            args=args,
            multimodal=True
        )

        print(f"Num negatives: {num_negatives}, Num positives: {num_positives}")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # --- Build models ---
        # Image model
        img_model = ImageModel(args).build_model().to(device)
        if args.img_checkpoint is not None:
            print("Loading image model pretrained weights...")
            image_checkpoint = os.path.join("/home/dvegaa/DELFOS/CARDIUM/img_script/image_checkpoints",
                                            args.img_checkpoint, f"fold{fold}_best_model.pth")
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
            tab_checkpoint = os.path.join("/home/dvegaa/DELFOS/CARDIUM/tabular_script/tabular_checkpoints",
                                          args.tab_checkpoint, f"best_model_fold_{fold+1}.pth")
            checkpoint = torch.load(tab_checkpoint, map_location=device, weights_only=True)
            tab_model.load_state_dict(checkpoint, strict=False)
        tab_model.mlp = nn.Identity()

        # Multimodal model
        multimodal_model = MultimodalModel(img_model=img_model, tab_model=tab_model, args=args).build_model().to(device)
        if args.multimodal_checkpoint is not None:
            print("Loading multimodal model pretrained weights...")
            multi_checkpoint = os.path.join("/home/dvegaa/DELFOS/CARDIUM/multimodal_script/multimodal_checkpoints",
                                            args.multimodal_checkpoint, f"fold{fold}_best_model.pth")
            checkpoint = torch.load(multi_checkpoint, map_location=device, weights_only=True)
            multimodal_model.load_state_dict(checkpoint, strict=False)

        # --- Loss function ---
        if args.loss_factor == 0:
            loss_weights = None
        else:
            loss_weights = torch.tensor([num_negatives / num_positives], dtype=torch.float32).to(device)
            if args.loss_factor > 1:
                loss_weights = torch.tensor(args.loss_factor).to(device)
            else:
                loss_weights = loss_weights * args.loss_factor

        print(f"Loss weights: {loss_weights}")
        criterion = nn.BCEWithLogitsLoss(pos_weight=loss_weights).to(device)

        # --- Optimizer ---
        optimizer = optim.AdamW(multimodal_model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)

        # --- Training Loop ---
        best_f1 = 0.0
        best_threshold = 0.5

        save_path = os.path.join("multimodal_checkpoints", exp_name, f"fold{fold}_best_model.pth")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        for epoch in range(args.num_epochs):
            print(f"Epoch [{epoch+1}/{args.num_epochs}]")

            # Train one epoch
            train_one_epoch(multimodal_model, train_loader, criterion, optimizer, epoch, device, fold, exp_name, args)

            # Validate
            best_threshold, best_f1, _, _ = evaluate(
                multimodal_model, test_loader, criterion, device, fold,
                mode="val", save_path=save_path, best_f1=best_f1, best_threshold=best_threshold
            )

            print(f"Best F1 so far: {best_f1}")


        # --- Test phase ---
        print("Loading the best model for test evaluation...")
        multimodal_model.load_state_dict(torch.load(save_path))
        f1_test = evaluate(multimodal_model, test_loader, criterion, device, fold, args, mode="test",
                           best_threshold=best_threshold)

        best_f1_folds.append(f1_test)

    # --- Summary ---
    mean_f1 = np.mean(best_f1_folds)
    std_f1 = np.std(best_f1_folds)
    print(f"Average F1-Score: {mean_f1:.4f} ± {std_f1:.4f}")

    wandb.log({"average_f1_score": mean_f1, "std_f1_score": std_f1})
    wandb.finish()


if __name__ == "__main__":
    main(args)
