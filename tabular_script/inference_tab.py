import torch
import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from data.tabular_data.load_tabular_data import TabularDataLoaders
from tabular_script.tab_models.get_tab_model import TabularModel
from utils import *

def main(args):
    # Prepare metrics storage
    metrics_folds ={"precision":[],
                    "recall": [],
                    "f1-score": [],
                    "accuracy": []}

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Cross-validation
    for fold in range(args.folds):
        print(f"\n=== Evaluating Fold {fold + 1} ===")
        set_seed(args.seed) 

        # Create data loaders for the current fold
        data_module = TabularDataLoaders(
            fold_idx=fold,
            base_dir=args.tab_output_dir,
            seed=args.seed,
            sampling=args.sampling,
            batch_size=args.batch_size
        )
        _, _, val_loader, _, _, _, _, _ = data_module.get_loaders()

        # Load multimodal model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Get tabular model 
        tab_model = TabularModel(args)
        tab_model = tab_model.build_model()
        tab_model = tab_model.to(device)

        tab_checkpoint = os.path.join(args.tab_checkpoint, f"fold{fold}_best_model.pth")
        tab_checkpoint = torch.load(tab_checkpoint, map_location = torch.device("cuda"), weights_only=True)
        tab_model.load_state_dict(tab_checkpoint, strict=False)
        tab_model = tab_model.to(device)

        # Inference
        y_true, y_score = inference_tabular_model(model=tab_model, loader=val_loader, device=device)
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