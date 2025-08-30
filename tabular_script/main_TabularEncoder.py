import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import wandb
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from tabular_script.preprocess import Clinical_Record_Preprocessing
from data.load_tabular_data import TabularDataLoaders
from tabular_script.tab_utils import *
from tabular_script.evaluate import evaluate_tabular_cv
from tabular_script.train import TabularTrainer
from tabular_script.tab_models.get_tab_model import TabularModel

args = get_main_parser()
exp_name = args.exp_name

exp_dir = os.path.join(args.weights_dir, args.exp_name)
os.makedirs(exp_dir, exist_ok=True)

config_path = os.path.join(exp_dir, "config.json")
with open(config_path, "w") as f:
    json.dump(vars(args), f, indent=4)

wandb.init(
    project="Final_Encoder_Reproducibility",
    name=exp_name,
    config=vars(args)
)

print("Preprocessing clinical records...")
preprocessed_data = Clinical_Record_Preprocessing(
    input_file=args.input_file,
    image_folder_path=args.image_folder_path,
    output_dir=args.output_dir,
    complete_output_dir=args.complete_output_dir
)
preprocessed_data.run()
print("Clinical records preprocessed successfully.")

exp_name = args.exp_name
wandb.init(
        project="Opt_pre_std",
        name=exp_name,
        config=vars(args)  
    )

set_seed(args.seed)
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

# Cargar folds y entrenar
fold_metrics = {'auc': [], 'precision': [], 'recall': [], 'f1': [], 'threshold': []}
all_val_probs, all_val_true = [], []

for fold_idx in range(args.folds):
    #fold_idx = 2  
    set_seed(args.seed)

    print(f"\n=== Fold {fold_idx + 1} ===")
    data_module = TabularDataLoaders(
        fold_idx=fold_idx,
        base_dir=args.output_dir,
        seed=args.seed,
        sampling=args.sampling,
        batch_size=args.batch_size
    )

    X_train_fold, train_loader, val_loader, positives, negatives, train_dataset, val_dataset, y_val_fold = data_module.get_loaders()


    model = TabularModel(args).build_model().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.sched_step, gamma=args.sched_gamma)

    loss_weights = torch.tensor([negatives / positives], dtype=torch.float32).to(device) * args.loss_weight_factor
    criterion = nn.BCEWithLogitsLoss(pos_weight=loss_weights)

    path_model = os.path.join(args.weights_dir, exp_name)

    trainer = TabularTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        train_dataset=train_dataset,
        batch_size=args.batch_size,
        seed=args.seed,
        positives=positives,
        weights_dir=path_model,
        fold_idx=fold_idx,
        hn_epochs=args.hn_epochs
    )

    best_f1_val, fold_val_probs = trainer.train(args.epochs)

    val_preds = (fold_val_probs > 0.5).astype(float)
    auc = roc_auc_score(y_val_fold, fold_val_probs)
    prec = precision_score(y_val_fold, val_preds, pos_label=1)
    rec = recall_score(y_val_fold, val_preds, pos_label=1)

    fold_metrics['auc'].append(auc)
    fold_metrics['precision'].append(prec)
    fold_metrics['recall'].append(rec)
    fold_metrics['f1'].append(best_f1_val)
    fold_metrics['threshold'].append(0.5)

    all_val_probs.extend(fold_val_probs)
    all_val_true.extend(y_val_fold)

    wandb.log({
        "fold": fold_idx + 1,
        "fold_auc": auc,
        "fold_precision": prec,
        "fold_recall": rec,
        "fold_f1": best_f1_val,
        "fold_threshold": 0.5,
    })

evaluate_tabular_cv(all_val_probs, all_val_true, fold_metrics)
wandb.finish() 