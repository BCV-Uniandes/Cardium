import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import wandb
import torch


def evaluate_tabular_cv(all_val_probs, all_val_true, fold_metrics, inference=False):
    """
    Evaluate cross-validation performance

    This function aggregates validation predictions from all folds in a 
    cross-validation experiment and reports aggregated metrics 
    across folds (AUC, precision, recall, F1). 
    Results are also logged to Weights & Biases (wandb).

    Args:
        all_val_probs:
            List/array of predicted probabilities across all validation folds.
        all_val_true:
            List/array of ground-truth binary labels (0 or 1) corresponding to `all_val_probs`.
        fold_metrics:
            Dictionary containing fold-level metrics for cross-validation.
            Expected keys: "auc", "precision", "recall", "f1".
    """
    all_val_probs = np.array(all_val_probs)
    all_val_true = np.array(all_val_true)

    print("\n=== Cross-Validation Results ===")
    print(f"AUC: {np.mean(fold_metrics['auc']):.4f} ± {np.std(fold_metrics['auc']):.4f}")
    print(f"Precision: {np.mean(fold_metrics['precision']):.4f} ± {np.std(fold_metrics['precision']):.4f}")
    print(f"Recall: {np.mean(fold_metrics['recall']):.4f} ± {np.std(fold_metrics['recall']):.4f}")
    print(f"F1-Score: {np.mean(fold_metrics['f1']):.4f} ± {np.std(fold_metrics['f1']):.4f}")
    if not inference:
        wandb.log({
            "AUC_mean": np.mean(fold_metrics['auc']),
            "AUC_std": np.std(fold_metrics['auc']),
            "precision_mean": np.mean(fold_metrics['precision']),
            "recall_mean": np.mean(fold_metrics['recall']),
            "f1_mean": np.mean(fold_metrics['f1']),
        })
    
def inference_fold(model, val_loader, device, all_val_probs, all_val_true, fold_idx):
    """
    Run validation for one fold, update global metrics, and print fold F1 score.
    Used for inference. 
    Args:
        model: Trained model to evaluate
        val_loader: Validation DataLoader for this fold
        device: Device ('cpu' or 'cuda')
        all_val_probs: list to accumulate validation probabilities across folds
        all_val_true: list to accumulate validation true labels across folds
        fold_metrics: dictionary to store fold-wise metrics (e.g., fold_metrics['f1'])
        fold_idx: current fold index (int)
    """
    model.eval()
    val_probs, val_true = [], []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = torch.sigmoid(model(inputs).squeeze())
            val_probs.extend(outputs.cpu().numpy())
            val_true.extend(targets.cpu().numpy())

    all_val_probs.extend(val_probs)
    all_val_true.extend(val_true)

    preds = (np.array(val_probs) > 0.5).astype(float)
    
    
    f1 = f1_score(val_true, preds)
    precision = precision_score(val_true, preds)
    recall = recall_score(val_true, preds)
    auc = roc_auc_score(val_true, val_probs)
    print(f"===Fold {fold_idx+1} results===")
    print(f"AUC = {auc:.4f}")
    print(f"Precision = {precision:.4f}")
    print(f"Recall = {recall:.4f}")
    print(f"F1-Score = {f1:.4f}")
    return auc, precision, recall, f1