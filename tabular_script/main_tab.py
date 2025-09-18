import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import wandb
import datetime
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from data.tabular_data.preprocess import Clinical_Record_Preprocessing
from data.tabular_data.load_tabular_data import TabularDataLoaders
from tabular_script.run_tab import TabularTrainer
from tabular_script.tab_models.get_tab_model import TabularModel
from utils import *

def main(args):
    """
    Main function to run K-Fold cross-validation training, validation, and testing
    for the tabular model.

    Args:
        args (Namespace): Parsed arguments containing all configurations.
    """
    # Initialize experiment in Weights & Biases
    exp_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    wandb.init(
        project="Final_Encoder_Reproducibility",
        entity="spared_v2", 
        name=exp_name
    )

    wandb.log({"args": vars(args)})

    ############## PROCESSING CLINICAL DATA ######################################
    print("Preprocessing clinical records...")
    preprocessed_data = Clinical_Record_Preprocessing(
        input_file=args.tab_input_file,
        image_folder_path=args.image_folder_path,
        output_dir=args.tab_output_dir,
        complete_output_dir=args.tab_complete_output_dir
    )
    preprocessed_data.run()
    print("Clinical records preprocessed successfully.")

    ######################### TRAIN MODEL ###########################################
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    folds = args.folds
    test_metrics = {"F1 Score": [],
                    "Accuracy": [], 
                    "Precision": [],
                    "Recall": []}

    for fold_idx in range(folds):
        print(f"\n=== Fold {fold_idx + 1} ===")

        # Set random seed for reproducibility
        set_seed(args.seed)

        # Create dataloaders
        data_module = TabularDataLoaders(
            fold_idx=fold_idx,
            base_dir=args.tab_output_dir,
            seed=args.seed,
            sampling=args.sampling,
            batch_size=args.batch_size
        )

        _, train_loader, val_loader, positives, negatives, train_dataset, _, _ = data_module.get_loaders()

        # --- Build model ---
        model = TabularModel(args).build_model().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.tab_sched_step, gamma=args.tab_sched_gamma)

        loss_weights = torch.tensor([negatives / positives], dtype=torch.float32).to(device) * args.loss_factor
        criterion = nn.BCEWithLogitsLoss(pos_weight=loss_weights)

        save_path = os.path.join("tabular_checkpoints", exp_name, f"fold{fold_idx}_best_model.pth")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # --- Initialize Trainer ---
        trainer = TabularTrainer(model=model,
                                args=args,
                                criterion=criterion,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                device=device,
                                train_loader=train_loader,
                                val_loader=val_loader,
                                train_dataset=train_dataset,
                                positives=positives,
                                fold_idx=fold_idx,
                                save_path=save_path,
                                hn_epochs=args.tab_hn_epochs
        )

        # Train model
        trainer.train(args.num_epochs)

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
    args = get_main_parser()
    main(args)