import torch.nn as nn
import torch
import torch.optim as optim
from tqdm import tqdm
import wandb
from datetime import datetime
import os
import pathlib
import sys

delfos_path = pathlib.Path(__name__).resolve().parent.parent
sys.path.append(str(delfos_path))
from data.transformations import transform_train, transform_test
from data.load_data_imgs import create_dataloaders
from img_script.train_img import train_one_epoch
from img_script.evaluate_img import evaluate
from img_script.get_img_model import ImageModel
from data.img_dataloader import DelfosDataset
from img_script.utils_imgs import *


def main(args): 
    # Track the arguments through wandb
    wandb.log({"args": vars(args),
            "model": args.img_model})

    # Save the best F1 scores for each fold
    best_f1_folds = []

    # K-Fold Cross Validation
    folds = args.folds
    for fold in range(folds):
        # Set the random seed for reproducibility
        set_seed(42)
        # Create Dataloaders for the current fold
        if args.trimester == None:
            dataset_path = f"{args.data_path}/fold_{fold+1}" # Without trimester separation
        else:
            dataset_path = f"{args.data_path}/{args.trimester}_trimester/fold_{fold+1}" # With trimester separation
            
        train_loader, val_loader, test_loader, num_negatives, num_positives, _ = create_dataloaders(
            dataset_dir = dataset_path, # Path to the dataset
            json_root="", # Only images are considered
            dataset_class=DelfosDataset, # Custom dataset class
            transform_train=transform_train, # Transformations for training data
            transform_test=transform_test, # Transformations for test data
            batch_size=args.batch_size, # Batch size
            fold=fold, # Current fold
            args=args # Additional arguments
        )
        print("Dataloaders have been successfully created")

        # Create the imag only model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = ImageModel(args)
        model = model.build_model()

        # Move the model to GPU (if available)
        model = model.to(device)

        # Calculate initial loss weights based on class imbalance
        loss_weights = torch.tensor([num_negatives / num_positives], dtype=torch.float32).to(device)
        # Adjust loss weights based on the provided loss factor
        if args.loss_factor == 0:
            loss_weights = None
        elif args.loss_factor >= 1:
            loss_weights = torch.tensor(args.loss_factor).to(device)
        else:
            loss_weights = loss_weights*args.loss_factor
        print(loss_weights)
        # Define loss function with the respective weights
        criterion = nn.BCEWithLogitsLoss(pos_weight=loss_weights)
        # Define optimizer
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)

        # Save best model
        save_path = os.path.join("image_checkpoints", exp_name, f"fold{fold}_best_model.pth")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Initialize the best F1 score 
        best_f1 = 0.0
        best_f1_count = 0.0
        counter = 0
        # Training and validation loop
        for epoch in tqdm(range(args.num_epochs)):
            print(f"Epoch [{epoch+1}/{args.num_epochs}]")

            # Training phase
            train_one_epoch(model, train_loader, criterion, optimizer, epoch, device, fold, args)

            # Validation phase
            best_f1, current_f1, val_loss = evaluate(model, test_loader, criterion, device, fold, args, mode="val", save_path=save_path, best_f1=best_f1) 
            
            # Check if the current F1 score is better than the best F1 score
            print(current_f1)
            if current_f1 >= best_f1_count:
                best_f1_count = current_f1
                counter = 0
            else:
                print("counter + 1")
                counter += 1
                print(counter)
                print(best_f1_count)
            
            print(f"best f1-score: {best_f1}")

        # Test phase with the best model
        # Test phase
        print("Loading the best model for test evaluation...")
        model.load_state_dict(torch.load(save_path))
        evaluate(model, test_loader, criterion, device, fold, args, mode="test")
        # Append the best F1 score for the current fold
        best_f1_folds.append(best_f1)

    # Calculate the average and standard deviation of the best F1 scores across all folds
    mean_f1 = np.mean(best_f1_folds)
    std_f1 = np.std(best_f1_folds)
    # Log the average and standard deviation of F1 scores to wandb
    wandb.log({"average_f1_score": mean_f1})
    wandb.log({"std_f1_score": std_f1})  
    # Print these values
    print(f"average f1 score: {mean_f1}")
    print(f"std f1 score: {std_f1}")  
    # Finish wandb logging
    wandb.finish()

if __name__ == "__main__":
    # Parse the arguments
    args = get_main_parser()
    # Set the experiment name 
    if args.exp_name is None:
        # Use the current date and time as the experiment name if not provided
        exp_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    else:
        # Use the provided experiment name
        exp_name = args.exp_name
    # Initialize wandb to track the experiment
    wandb.init(project=args.wandb_project, 
            name=exp_name)
    # Start the main function
    main(args)