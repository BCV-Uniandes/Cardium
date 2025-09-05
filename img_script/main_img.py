import torch.nn as nn
import torch
import torch.optim as optim
from tqdm import tqdm
import wandb
import datetime
import os
import pathlib
import sys

delfos_path = pathlib.Path(__name__).resolve().parent.parent
sys.path.append(str(delfos_path))
from data.transformations import transform_train, transform_test
from data.load_img_data import create_dataloaders
from img_script.train_img import train_one_epoch
from img_script.evaluate_img import evaluate
from img_script.img_models.get_img_model import ImageModel
from data.img_dataloader import DelfosDataset
from data.dataloader import CardiumDataset
from utils import *
from img_script.run import ImageTrainer

# Parse the arguments
args = get_main_parser()

def main(args): 
    # Initialize experiment in Weights & Biases
    exp_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    wandb.init(project="MedVit", 
            entity="spared_v2", 
            name=exp_name)

    wandb.log({"args": vars(args)})

    folds = args.folds
    test_metrics = {"F1 Score": [],
                    "Accuracy": [], 
                    "Precision": [],
                    "Recall": []}
    
    for fold in range(folds):
        # Set the random seed for reproducibility
        set_seed(42)
        
        # Create Dataloaders for the current fold
        dataset_path = f"{args.image_folder_path}/fold_{fold+1}" # Without trimester separation

        train_loader, test_loader = create_dataloaders(
            dataset_dir = dataset_path,
            dataset_class=CardiumDataset, 
            transform_train=transform_train, 
            transform_test=transform_test, 
            batch_size=args.batch_size,
            args=args 
        )
        print("Dataloaders have been successfully created")

        # Create the imag only model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = ImageModel(args)
        model = model.build_model()

        # Move the model to GPU (if available)
        model = model.to(device)

        # --- Loss function ---
        loss_weights = torch.tensor(args.loss_factor).to(device)

        print(f"Loss weights: {loss_weights}")
        criterion = nn.BCEWithLogitsLoss(pos_weight=loss_weights).to(device)

        # --- Optimizer ---
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)

        # --- Training Loop ---
        # Save best model
        save_path = os.path.join("image_checkpoints", exp_name, f"fold{fold}_best_model.pth")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        trainer = ImageTrainer(model, 
                               args, 
                               criterion, 
                               optimizer, 
                               device, 
                               train_loader, 
                               test_loader, 
                               fold, 
                               save_path)

        # Training and validation loop
        for epoch in tqdm(range(args.num_epochs)):
            print(f"Epoch [{epoch+1}/{args.num_epochs}]")

            # Training phase
            trainer.train_one_epoch(epoch)

            # Validation phase
            trainer.validate_one_epoch()

        # --- Test phase ---
        print("Loading the best model for test evaluation...")
        model.load_state_dict(torch.load(save_path))                                                    
        f1_test, accuracy_test, precision_test, recall_test = trainer.test_one_epoch(model)

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