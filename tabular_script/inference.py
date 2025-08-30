import torch
import os
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.load_tabular_data import TabularDataLoaders
from tabular_script.tab_models.get_tab_model import TabularModel
from tabular_script.evaluate import *
from tabular_script.tab_utils import *

args = get_main_parser()

weights_dir = args.weights_dir
n_folds = args.folds
embedding_dim = args.tab_feature_dim
num_heads = args.tab_num_heads
num_layers = args.tab_num_layers
batch_size = args.batch_size
device = args.device
base_dir = args.output_dir
seed=args.seed
exp_name=args.exp_name

exp_dirs = [os.path.join(weights_dir, d) for d in os.listdir(weights_dir)]
exp_dir = max(exp_dirs, key=os.path.getmtime)
config_path = os.path.join(exp_dir, "config.json")
with open(config_path, "r") as f:
    config = json.load(f)
exp_name = config["exp_name"]  

fold_metrics = {
    'auc': [], 'precision': [], 'recall': [], 'f1': [], 'threshold': []
}
all_val_probs = []
all_val_true = []

for fold_idx in range(n_folds):
    print(f"\n=== Evaluating Fold {fold_idx + 1} ===")
    _, _, X_val_fold, y_val_fold = load_fold(fold_idx, base_dir, seed)

    data_module = TabularDataLoaders(
        fold_idx=fold_idx,
        base_dir=args.output_dir,
        seed=args.seed,
        sampling=args.sampling,
        batch_size=args.batch_size
    )
    _, _, val_loader, _, _, _, _, _ = data_module.get_loaders()

    model = TabularModel(args).build_model().to(device)

    fold_weights_path = os.path.join(weights_dir, exp_name, f"best_model_fold_{fold_idx + 1}.pth")
    if os.path.exists(fold_weights_path):
        model.load_state_dict(torch.load(fold_weights_path, map_location=device))
        print(f"Weights loaded for fold {fold_idx+1} from: {fold_weights_path}")
    else:
        print(f"Weight file not found for fold {fold_idx+1}.")
        continue

    auc, precision, recall, f1 = inference_fold(
        model=model,
        val_loader=val_loader,
        device=args.device,
        all_val_probs=all_val_probs,
        all_val_true=all_val_true,
        fold_idx=fold_idx
    )

    fold_metrics['auc'].append(auc)
    fold_metrics['precision'].append(precision)
    fold_metrics['recall'].append(recall)
    fold_metrics['f1'].append(f1)

evaluate_tabular_cv(all_val_probs, all_val_true, fold_metrics, inference=True)
