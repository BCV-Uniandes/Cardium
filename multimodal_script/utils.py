import argparse
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score, classification_report
from tqdm import tqdm
import torch
from tqdm import tqdm
import torch
import numpy as np
import random
from collections import defaultdict
import json
import wandb

str2bool = lambda x: (str(x).lower() == 'true')

def get_main_parser():
    parser = argparse.ArgumentParser(description="Training configuration for Vision Transformer")

    # Add arguments
    ######################################## MULTIMODAL MODEL #############################################################################
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-7, help="Learning rate for optimizer")
    parser.add_argument("--n_classes", type=int, default=1, help="Number of output classes")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Embedding dimension")
    parser.add_argument("--loss_factor", type=float, default=1.2, help="factor to multiply weight_loss")
    parser.add_argument("--multimodal_pretrain", type=str2bool, default=False, help="True for pretrained multimodal model")
    parser.add_argument("--multimodal_model", type=str, default="TransDoubleCross", help="whether to use MLP (mlp), Transformer Encoder (TransEncoder), Transformer Decoder (TransDecoder), Single cross attention (TransCross) or unimodal (unimodal) model")
    parser.add_argument("--multimodal_checkpoint", type=str, default=None, help="Multimodal checkpoint for finutenning")
    parser.add_argument("--img_feature_dim", type=int, default=384, help="Dimension of image features to be used in multimodal model")
    parser.add_argument("--embed_dim", type=int, default=384, help="Embedding dimensions of the TransformerEncoder model")
    parser.add_argument("--num_heads", type=int, default=2, help="Number of heads of the TransformerEncoder model")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of layers of the TransformerEncoder model")
    parser.add_argument("--folds", type=int, default=3, help="Number of folds in k-fold cross validation")
    parser.add_argument("--fold", type=int, default=0, help="Fold to use for experimentation")
    parser.add_argument("--path_dropout", type=float, default=0.4, help="Path dropout to be used in multimodal models")
    parser.add_argument("--class_dropout", type=float, default=0.4, help="Path dropout to be used in multimodal models")
    parser.add_argument("--exp_name", type=str, default="", help="Experiment name for optimization threshold")
    parser.add_argument("--sampling", type=str2bool, default=True, help="Whether to use weighted random sampling in dataloader or not")
    parser.add_argument("--smooth_label", type=str2bool, default=False, help="Whether to use smooth label or not")
    parser.add_argument("--frozen", type=str2bool, default=True, help="Whether to use frozen encoders weights or not")
    parser.add_argument("--unimodal", type=str, default="img", help="Whether to use image modality (img) or tabular modality (tab)")
    parser.add_argument("--trimester", type=str, default=None, help="Which trimester to evaluate")
    ########################################## IMAGE MODEL ###############################################################################
    parser.add_argument("--img_pretrain", type=str2bool, default=True, help="True for pretrained image model")
    parser.add_argument("--img_model", type=str, default="vit_small", help="whether to use medvit, vit_tiny, vity_small, resnet18, resnet50")
    parser.add_argument("--img_checkpoint", type=str, default="vit_small", help="Path to image model checkpoint")
    parser.add_argument("--img_path_dropout", type=float, default=0.3, help="Path dropout to be used in multimodal models")
    parser.add_argument("--img_class_dropout", type=float, default=0.2, help="Path dropout to be used in multimodal models")
    ######################################### TABULAR MODEL ##############################################################################
    parser.add_argument("--tab_pretrain", type=str2bool, default=False, help="True for pretrained tabular model")
    parser.add_argument("--tab_model", type=str, default="TabEncoder", help="whether to use TabTransformer")
    parser.add_argument("--tab_feature_dim", type=int, default=128, help="Dimension of tabular features to be used in multimodal model")
    parser.add_argument("--tab_checkpoint", type=str, default="best_tab", help="Path to tabular model checkpoint")
    parser.add_argument("--tab_num_heads", type=int, default=8, help="Number of heads of the TransformerEncoder model")
    parser.add_argument("--tab_num_layers", type=int, default=2, help="Number of layers of the TransformerEncoder model")
    parser.add_argument("--tab_num_features", type=int, default=97, help="The dimension of the input vector")
    return parser.parse_args()

def set_seed(seed):
    """
    Set the seed for reproducibility.

    Args:
        seed (int): The seed value.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def inference_multimodal(model, loader, device, threshold=0.5):
    """
    Evaluate the model on the given data loader and calculate metrics.

    Args:
        model (torch.nn.Module): The model to evaluate.
        loader (torch.utils.data.DataLoader): DataLoader for the evaluation dataset.
        device (torch.device): Device to perform computations on (e.g., 'cuda' or 'cpu').
        threshold (float): Threshold for binary classification.

    Returns:
        tuple: Precision, recall, accuracy, F1-score, predicted scores (y_score), and true labels (y_true).
    """
    y_true_patient = {}
    y_score_patient = defaultdict(list)

    model.eval()
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Evaluating"):
            img_data, tab_data, patient_ids = inputs[0].to(device), inputs[1].to(device), inputs[2]
            targets = targets.to(device)

            # Forward pass
            outputs = model(img_data, tab_data).squeeze(1)
            targets = targets.to(torch.float32)

            # Sigmoid to convert logits to probabilities
            probabilities = torch.sigmoid(outputs).cpu().numpy()
            targets = targets.cpu().numpy()

            # Aggregate scores and true labels by patient ID
            for pid, prob, target in zip(patient_ids, probabilities, targets):
                y_score_patient[pid].append(prob)
                y_true_patient[pid] = target  # True label is same for all occurrences of patient
 

    # Average scores for each patient
    y_score_patient_avg = {pid: sum(scores) / len(scores) for pid, scores in y_score_patient.items()}
    y_true_patient = {pid: int(label) for pid, label in y_true_patient.items()}

    # Convert to arrays for metric computation
    y_true = list(y_true_patient.values())
    y_score = list(y_score_patient_avg.values())

    return y_true, y_score

def compute_patient_metrics(
    y_score_patient: dict, 
    y_true_patient: dict, 
    mode: str = "val", 
    fold: str = None, 
    threshold: float = 0.5
):
    """
    Compute patient-level metrics by averaging prediction scores per patient.

    Args:
        y_score_patient (dict): Dictionary mapping patient IDs to a list of prediction scores.
        y_true_patient (dict): Dictionary mapping patient IDs to true labels (0 or 1).
        mode (str): Evaluation mode, "val" or "test". Determines logging behavior.
        fold (str, optional): Fold identifier for test metrics logging.
        best_threshold (float): Threshold to convert probabilities into binary predictions.

    Returns:
        f1 (float): Patient-level F1 score.
        best_threshold (float): Threshold used for predictions.
    """

    # --- 1. Average scores per patient ---
    y_score_patient_avg = {pid: np.mean(scores) for pid, scores in y_score_patient.items()}

    # Ensure true labels are aligned with patient IDs
    y_true_patient = {pid: label for pid, label in y_true_patient.items()}

    # --- 2. Convert dictionaries to arrays for metric computation ---
    y_true = np.array(list(y_true_patient.values()))
    y_score = np.array(list(y_score_patient_avg.values()))

    # --- 3. Apply threshold to obtain binary predictions ---
    y_pred = (y_score > threshold).astype(int)

    # --- 4. Compute common classification metrics ---
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # --- 5. Log and print patient-level metrics ---
    print(f"{mode} Metrics (Patient-level) on fold {fold}: "
          f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    wandb.log({
        f"{mode.lower()}_{fold}_accuracy_patient": accuracy,
        f"{mode.lower()}_{fold}_precision_patient": precision,
        f"{mode.lower()}_{fold}_recall_patient": recall,
        f"{mode.lower()}_{fold}_f1_score_patient": f1,
    })

    # --- 6. Additional metrics for test mode ---
    if mode.lower() == "test" and fold is not None:
        # Print classification report per class
        print("Classification Report for fold {fold}:\n", classification_report(
            y_true, y_pred, target_names=["No Cardiopatia", "Cardiopatia"]
        ))

    return f1, accuracy, precision, recall

def save_predictions(model, loader, device, fold, threshold=0.5):
    """
    Evaluate the model on the given data loader and calculate metrics.

    Args:
        model (torch.nn.Module): The model to evaluate.
        loader (torch.utils.data.DataLoader): DataLoader for the evaluation dataset.
        device (torch.device): Device to perform computations on (e.g., 'cuda' or 'cpu').
        threshold (float): Threshold for binary classification.

    Returns:
        tuple: Precision, recall, accuracy, F1-score, predicted scores (y_score), and true labels (y_true).
    """
    y_true_patient = {}
    y_score_patient = defaultdict(list)
    predicciones_list = []

    model.eval()
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Evaluating"):
            img_data, tab_data, patient_ids, img_paths = inputs[0].to(device), inputs[1].to(device), inputs[2], inputs[3]
            targets = targets.to(device)

            # Forward pass
            outputs = model(img_data, tab_data).squeeze(1)
            targets = targets.to(torch.float32)

            # Sigmoid to convert logits to probabilities
            probabilities = torch.sigmoid(outputs).cpu().numpy()
            targets = targets.cpu().numpy()

            # Group predictions and true labels by patient_id
            for patient_id, img_path, prob, target in zip(patient_ids, img_paths, probabilities, targets):
                if patient_id not in y_score_patient:
                    y_score_patient[patient_id] = np.array([prob])  # Initialize as array
                else:
                    y_score_patient[patient_id] = np.append(y_score_patient[patient_id], prob)  # Append to array

                y_true_patient[patient_id] = target  

                predicciones_list.append({
                    "ID": patient_id,
                    "y_true": int(target),
                    "y_pred": float(prob),
                    "img_name": img_path.split("/")[-1]
                })

    correct_ids = []
    incorrect_ids = []
    patient_ids_list = list(y_true_patient.keys())
    # Average scores for each patient
    y_score_patient_avg = {pid: sum(scores) / len(scores) for pid, scores in y_score_patient.items()}
    y_true_patient = {pid: int(label) for pid, label in y_true_patient.items()}

    # Convert to arrays for metric computation
    y_true = list(y_true_patient.values())
    y_score = list(y_score_patient_avg.values())
    y_pred = [1 if score > threshold else 0 for score in y_score]  # Apply threshold of 0.5

    for i in range(len(y_pred)):
        if y_true[i] == 1:
            if y_pred[i] == y_true[i]:
                correct_ids.append(patient_ids_list[i])
            else:
                incorrect_ids.append(patient_ids_list[i])
    
    # Calculate metrics
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)

    with open(f"/home/dvegaa/DELFOS/CARDIUM/multimodal_script/predictions/predicciones_pacientes_fold_{str(fold)}.json", "w") as f:
        json.dump(predicciones_list, f, indent=4)
    
    return precision, recall, accuracy, f1, roc_auc, y_score, y_true

