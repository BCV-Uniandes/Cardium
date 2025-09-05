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
import pathlib
from sklearn.model_selection import KFold
import copy
import os

str2bool = lambda x: (str(x).lower() == 'true')

def get_main_parser():
    parser = argparse.ArgumentParser(description="Training configuration for Vision Transformer")
    BASE_DIR = pathlib.Path(__name__).resolve().parent.parent
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
    parser.add_argument("--sampling", type=str2bool, default=True, help="Whether to use weighted random sampling in dataloader or not")
    parser.add_argument("--unimodal", type=str, default="img", help="Whether to use image modality (img) or tabular modality (tab)")
    parser.add_argument("--image_folder_path", type=str, default=str("/home/dvegaa/DELFOS/CARDIUM/anon_dataset"), help="Image data path for training and evaluation")
    parser.add_argument("--json_path", type=str, default=None, help="Path to json file containing tabular data for training and evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    ########################################## IMAGE MODEL ###############################################################################
    parser.add_argument("--img_pretrain", type=str2bool, default=True, help="True for pretrained image model")
    parser.add_argument("--img_model", type=str, default="vit_small", help="whether to use medvit, vit_tiny, vity_small, resnet18, resnet50")
    parser.add_argument("--img_checkpoint", type=str, default="vit_small", help="Path to image model checkpoint")
    parser.add_argument("--img_path_dropout", type=float, default=0.3, help="Path dropout to be used in multimodal models")
    parser.add_argument("--img_class_dropout", type=float, default=0.2, help="Path dropout to be used in multimodal models")
    ######################################### TABULAR MODEL ##############################################################################
    parser.add_argument("--tab_pretrain", type=str2bool, default=False, help="True for pretrained tabular model")
    parser.add_argument("--tab_model", type=str, default="TabTransformer", help="whether to use TabTransformer")
    parser.add_argument("--tab_feature_dim", type=int, default=128, help="Dimension of tabular features to be used in multimodal model")
    parser.add_argument("--tab_checkpoint", type=str, default="best_tab", help="Path to tabular model checkpoint")
    parser.add_argument("--tab_num_heads", type=int, default=8, help="Number of heads of the TransformerEncoder model")
    parser.add_argument("--tab_num_layers", type=int, default=2, help="Number of layers of the TransformerEncoder model")
    parser.add_argument("--tab_num_features", type=int, default=97, help="The dimension of the input vector")
    parser.add_argument("--tab_hn_epochs", type=int, default=20, help="Frequency (in epochs) for Hard Negative Mining")  
    parser.add_argument("--tab_sched_step", type=int, default=100, help="Step size for the learning rate scheduler")  
    parser.add_argument("--tab_sched_gamma", type=float, default=0.6, help="Gamma factor for the scheduler") 
    ################################## TABULAR DATA PREPROCESSING ##########################################################################
    parser.add_argument("--tab_input_file", type=str, default=str(BASE_DIR/"data/tabular_data/delfos_clinical_data_wnm_translated_final_cleaned.json"))
    parser.add_argument("--tab_output_dir", type=str, default=str(BASE_DIR/"data/tabular_data/output_folds_final"))
    parser.add_argument("--tab_complete_output_dir", type=str, default=str(BASE_DIR/"data/tabular_data"))

    return parser.parse_args()

############################ REPRODUCIBILITY ###########################################################################
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

def worker_init_fn(worker_id, seed):
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)

################################ METRICS #########################################################
def compute_patient_metrics(
    y_score: np.array, 
    y_true: np.array, 
    mode: str = "val", 
    fold: str = None, 
    threshold: float = 0.5,
    log_wandb: bool = True
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

    # --- 1. Apply threshold to obtain binary predictions ---
    y_pred = (y_score > threshold).astype(int)

    # --- 2. Compute common classification metrics ---
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # --- 3. Log and print patient-level metrics ---
    print(f"{mode} Metrics (Patient-level) on fold {fold}: "
          f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    if log_wandb:
        wandb.log({
            f"{mode.lower()}_accuracy_patient": accuracy,
            f"{mode.lower()}_precision_patient": precision,
            f"{mode.lower()}_recall_patient": recall,
            f"{mode.lower()}_f1_score_patient": f1,
        })

    # --- 4. Additional metrics for test mode ---
    if mode.lower() == "test" and fold is not None:
        # Print classification report per class
        print(f"Classification Report for fold {fold}:\n", classification_report(
            y_true, y_pred, target_names=["No Cardiopatia", "Cardiopatia"]
        ))

    return f1, accuracy, precision, recall

######################## PREPROCESSING FUNCTIONS ################################
def calculate_woe(data, feature, target):
    """Estimates the Weight of Evidence (WoE) for a fold of a categorical
    feature.
    Args:
    data (pd.DataFrame): DataFrame containing the feature and target.
    feature (str): Name of the categorical feature to encode.
    target (str): Name of the target variable."""
    eps = 1e-7  
    grouped = data.groupby(feature)[target].agg(['count', 'sum'])
    grouped['non_event'] = grouped['count'] - grouped['sum']
    grouped['event_rate'] = (grouped['sum'] + eps) / grouped['sum'].sum()
    grouped['non_event_rate'] = (grouped['non_event'] + eps) / grouped['non_event'].sum()
    grouped['woe'] = np.log(grouped['event_rate'] / grouped['non_event_rate'])
    return grouped['woe'].to_dict()

def woe_encoder_list(data, feature, target, max_len, n_splits=5):
    """Applies Weight of Evidence (WoE) encoding to categorical features, following a 5 cross encoding strategy. 
    Args: 
    data (pd.DataFrame): DataFrame containing the feature and target.
    feature (str): Name of the categorical feature to encode.
    target (str): Name of the target variable.
    max_len (int): Maximum length of the WoE vector for padding.
    n_splits (int): Number of splits for cross-encoding (default is 5).
    Returns:
    data_encoded (pd.DataFrame): DataFrame with the WoE encoded feature."""
    skf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    data_encoded = data.copy()
    data_encoded[f"{feature}_woe"] = [[] for _ in range(len(data))]

    for train_index, val_index in skf.split(data, data[target]):
        train_data = data.iloc[train_index]
        val_data = data.iloc[val_index]

        train_data_exploded = train_data.explode(feature)
        val_data_exploded = val_data.explode(feature)

        woe_map = calculate_woe(train_data_exploded, feature, target)

        val_data_exploded[f"{feature}_woe"] = val_data_exploded[feature].map(woe_map).fillna(0)

        val_data_woe = (
            val_data_exploded.groupby(val_data_exploded.index)[f"{feature}_woe"]
            .apply(list)
            .reindex(val_data.index)
        )

        for idx, vector in zip(val_index, val_data_woe):
            padded_vector = vector + [0] * (max_len - len(vector))
            data_encoded.at[idx, f"{feature}_woe"] = padded_vector

    return data_encoded

def normalize_categorical_woe(data, categorical_features):
    """
    Standardizes encoded features (WoE or similar) to have mean 0 and standard deviation 1.
    Ignores -1 values and preserves their positions.

    Parameters:
    - data: DataFrame containing the features (pandas DataFrame).
    - categorical_features: List of column names (WoE features) to standardize.

    Returns:
    - New DataFrame with standardized WoE features.
    """
    transformed_data = copy.deepcopy(data)

    for feature in categorical_features:
        raw_values = []

        for val in transformed_data[feature]:
            if isinstance(val, list):
                raw_values.extend([v for v in val if v != -1])
            elif isinstance(val, (int, float)) and val != -1:
                raw_values.append(val)

        if not raw_values:
            continue

        raw_values = np.array(raw_values)
        mean = raw_values.mean()
        std = raw_values.std() if raw_values.std() > 0 else 1.0  
        standardized_values = ((raw_values - mean) / std).flatten()

        idx = 0
        for i in range(len(transformed_data)):
            val = transformed_data.at[i, feature]
            if isinstance(val, list):
                new_list = []
                for v in val:
                    if v == -1:
                        new_list.append(v)
                    else:
                        new_list.append(standardized_values[idx])
                        idx += 1
                transformed_data.at[i, feature] = new_list
            elif isinstance(val, (int, float)) and val != -1:
                transformed_data.at[i, feature] = standardized_values[idx]
                idx += 1

    return transformed_data
    
def pad_categorical_lists(df, categorical_features, max_lengths):
    """Fills categorical features vectors with 'NA' to ensure that features 
     from all patients have the same length.
    Args:
    df (pd.DataFrame): DataFrame containing tabular data.
    categorical_features (list): List of categorical features to pad.
    max_lengths (list): List of maximum lengths for each categorical feature.
    Returns:
    pd.DataFrame: DataFrame with padded categorical features."""
    df_padded = df.copy()
    for feature, max_len in zip(categorical_features, max_lengths):
        df_padded[feature] = df_padded[feature].apply(
            lambda x: x + ["NA"] * (max_len - len(x)) if isinstance(x, list) else ["NA"] * max_len
        )
    return df_padded
        
def pad_numeric_features(row, numeric_features, max_lengths):
    """Fills numeric features vectors with -1 to ensure that features
    from all patients have the same length.
    Args:
    row (pd.Series): Row of the DataFrame containing patient features.
    numeric_features (list): List of numeric features to pad.
    max_lengths (list): List of maximum lengths for each numeric feature.
    Returns:
    pd.Series: Row with padded numeric features for a specific patient."""
    for feature, max_len in zip(numeric_features, max_lengths):
        if feature in row:
            value = row[feature]
            if not isinstance(value, list):
                value = [value]
            padded_value = value + [-1] * (max_len - len(value))
            row[feature] = padded_value[:max_len] 
    return row

def standardize_data(data, numeric_features, numerical_values):
    """We apply z-score normalization to the data, ignoring -1 values (missing values).
    Args:
        data (list): List of dictionaries containing the data.
        numeric_features (list): List of numeric features to standardize.
        numerical_values (list): List of individual numeric values to standardize (this includes
        unique integers and ordinal categorical variables)
    Returns:
        standardized_data (list): Data (list of dictionaries) with standardized values."""
    standardized_data = copy.deepcopy(data)

    for feature in numeric_features:
        values = []
        for item in data:
            if feature in item:
                if isinstance(item[feature], list):
                    values.extend([value for value in item[feature] if value != -1])
                elif isinstance(item[feature], (int, float)) and item[feature] != -1:
                    values.append(item[feature])
        if values:
            mean = np.mean(values)
            std = np.std(values)
            for item in standardized_data:
                if feature in item:
                    if isinstance(item[feature], list):
                        item[feature] = [
                            (value - mean) / std if value != -1 else -1 
                            for value in item[feature]
                        ]
                    elif isinstance(item[feature], (int, float)) and item[feature] != -1:
                        item[feature] = (item[feature] - mean) / std

    for feature in numerical_values:
        values = [item[feature] for item in data if feature in item and item[feature] != -1]
        if values:
            mean = np.mean(values)
            std = np.std(values)
            for item in standardized_data:
                if feature in item and item[feature] != -1:
                    item[feature] = (item[feature] - mean) / std

    return standardized_data

def create_folds(data, folder_path, file_names):
    """Creates a list for each fold with tabular data corresponding to the images previpously divided in folds.
    Args:
        data (list): List of dictionaries containing the data.
        folder_path (str): Path to thefolder containing the images previously divided in folds. 
        file_names (list): List of file names for localizing the images corresonding to each fold.
    Returns:
        list with tabular data for each fold."""
    folds = {}
    for i in file_names:
        files_path = os.path.join(folder_path, i)
        path_id_train_1 = os.path.join(files_path, 'train', 'Cardiopatia')
        path_id_train_2 = os.path.join(files_path, 'train', 'No_Cardiopatia')
        path_id_test_1 = os.path.join(files_path, 'test', 'Cardiopatia')
        path_id_test_2 = os.path.join(files_path, 'test', 'No_Cardiopatia')

        id_train = [f for f in os.listdir(path_id_train_1)] + [f for f in os.listdir(path_id_train_2)]
        id_test = [f for f in os.listdir(path_id_test_1)] + [f for f in os.listdir(path_id_test_2)]

        folds[f'{i}_train'] = id_train
        folds[f'{i}_test'] = id_test

    fold_1_train, fold_1_test = [], []
    fold_2_train, fold_2_test = [], []
    fold_3_train, fold_3_test = [], []

    for i in data:
        if i['id'] in folds['fold_1_train']:
            fold_1_train.append(i)
        if i['id'] in folds['fold_1_test']:
            fold_1_test.append(i)
        if i['id'] in folds['fold_2_train']:
            fold_2_train.append(i)
        if i['id'] in folds['fold_2_test']:
            fold_2_test.append(i)
        if i['id'] in folds['fold_3_train']:
            fold_3_train.append(i)
        if i['id'] in folds['fold_3_test']:
            fold_3_test.append(i)

    return fold_1_train, fold_1_test, fold_2_train, fold_2_test, fold_3_train, fold_3_test

######################### INFERENCE FUNCTIONS ########################################
def inference_multimodal(model, loader, device):
    """
    Evaluate the multimodal model on the given data loader and calculate metrics.

    Args:
        model (torch.nn.Module): The model to evaluate.
        loader (torch.utils.data.DataLoader): DataLoader for the evaluation dataset.
        device (torch.device): Device to perform computations on (e.g., 'cuda' or 'cpu').
        
    Returns:
        tuple: True labels (y_true) and predicted scores (y_score).
    """
    y_true_patient = {}
    y_score_patient = defaultdict(list)

    # Evaluate the model on the validation or test set
    model.eval()
    
    # --- 1. Iterate over DataLoader ---
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Evaluating"):
            # Move inputs and targets to the appropriate device
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
    
    # --- 2. Average scores per patient ---
    y_score_patient_avg = {pid: np.mean(scores) for pid, scores in y_score_patient.items()}
    y_true_patient = {pid: label for pid, label in y_true_patient.items()}

    # --- 3. Convert dictionaries to arrays for metric computation ---
    y_true = np.array(list(y_true_patient.values()))
    y_score = np.array(list(y_score_patient_avg.values()))


    return y_true, y_score

def inference_image_model(model, loader, device):
    """
    Evaluate the image model on the given data loader and calculate metrics.

    Args:
        model (torch.nn.Module): The model to evaluate.
        loader (torch.utils.data.DataLoader): DataLoader for the evaluation dataset.
        device (torch.device): Device to perform computations on (e.g., 'cuda' or 'cpu').
    
    Returns:
        tuple: True labels (y_true) and predicted scores (y_score).
    """
    y_true_patient = {}
    y_score_patient = defaultdict(list)

    # Evaluate the model on the validation or test set
    model.eval()
    
    # --- 1. Iterate over DataLoader ---
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Evaluating"):
            # Move inputs and targets to the appropriate device
            inputs, patient_ids = inputs[0].to(device), inputs[1]
            targets = targets.unsqueeze(1).to(device)
            targets = targets.to(torch.float32)
            
            # Forward pass
            outputs = model(inputs)
    
            # Convert logits to probabilities
            probabilities = torch.sigmoid(outputs).cpu().numpy()
            targets_np = targets.cpu().numpy()

            # Aggregate scores and true labels by patient ID
            for pid, prob, target in zip(patient_ids, probabilities, targets_np):
                y_score_patient[pid].append(prob)
                y_true_patient[pid] = target  # True label is same for all occurrences of patient

    # --- 2. Average scores per patient ---
    y_score_patient_avg = {pid: np.mean(scores) for pid, scores in y_score_patient.items()}
    y_true_patient = {pid: label for pid, label in y_true_patient.items()}

    # --- 3. Convert dictionaries to arrays for metric computation ---
    y_true = np.array(list(y_true_patient.values()))
    y_score = np.array(list(y_score_patient_avg.values()))

    return y_true, y_score

def inference_tabular_model(model, loader, device):
    """
    Evaluate the tabular model on the given data loader and calculate metrics.

    Args:
        model (torch.nn.Module): The model to evaluate.
        loader (torch.utils.data.DataLoader): DataLoader for the evaluation dataset.
        device (torch.device): Device to perform computations on (e.g., 'cuda' or 'cpu').
    
    Returns:
        tuple: True labels (y_true) and predicted scores (y_score).
    """
    model.eval()
    y_score, y_true = [], []
    # --- 1. Iterate over DataLoader ---
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).squeeze()
            outputs_prob = torch.sigmoid(outputs)
            y_score.extend(outputs_prob.cpu().numpy())
            y_true.extend(targets.cpu().numpy())

    # --- 2. Convert lists to arrays for metric computation ---
    y_score = np.array(y_score)
    y_true = np.array(y_true)

    return y_true, y_score
