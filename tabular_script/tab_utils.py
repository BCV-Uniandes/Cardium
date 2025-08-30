import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import os
import copy
import pandas as pd
import torch
import random
from datetime import datetime
import copy
import argparse
from pathlib import Path

###########ARGUMENTS PARSER############
def get_main_parser():
    parser = argparse.ArgumentParser(description="Training TabTransformer with Cross-Validation")
    #BASE_DIR = Path(__file__).resolve().parent.parent Actualizar cuando subamos los datos
    BASE_DIR = Path("/home/hceballos/anaconda3/CARDIUM")
    
    # Tabular Data Preprocessing
    parser.add_argument("--input_file", type=str, default=str(BASE_DIR / "data/TabularData/delfos_clinical_data_wnm_translated_final_cleaned.json"))
    parser.add_argument("--output_dir", type=str, default=str(BASE_DIR / "data/TabularData/output_folds_final"))
    parser.add_argument("--complete_output_dir", type=str, default=str(BASE_DIR / "data/TabularData"))
    parser.add_argument("--image_folder_path", type=str, default=str(BASE_DIR/"data/anon_dataset"), help="Path to the folder containing image IDs") 
    # Tabular Encoder & Training Parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda", help="Training device: 'cuda' or 'cpu'")
    parser.add_argument("--exp_name", type=str, default=datetime.now().strftime("%Y-%m-%d-%H-%M-%S"), help="Experiment name")
    parser.add_argument("--folds", type=int, default=3, help="Number of folds for cross-validation") 
    parser.add_argument("--lr", type=float, default=0.00000050169647031011, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=400, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--tab_num_heads", type=int, default=8, help="Number of attention heads in the transformer") 
    parser.add_argument("--tab_feature_dim", type=int, default=128, help="Dimension size of embeddings")  
    parser.add_argument("--tab_num_layers", type=int, default=2, help="Number of transformer layers")  
    parser.add_argument("--hn_epochs", type=int, default=20, help="Frequency (in epochs) for Hard Negative Mining")  
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="Weight decay for Adam optimizer")
    parser.add_argument("--sched_step", type=int, default=100, help="Step size for the learning rate scheduler")  
    parser.add_argument("--sched_gamma", type=float, default=0.6, help="Gamma factor for the scheduler")  
    parser.add_argument("--loss_weight_factor", type=float, default=0.7, help="Weight factor for the loss function")
    parser.add_argument("--sampling", type=int, default=0, help="Multiplier for the number of positives in sampling")
    parser.add_argument("--weights_dir", type=str, default=str(BASE_DIR / "tabular_checkpoints"))
    parser.add_argument("--tab_model", type=str, default=str("TabTransformer"), help="Tabular model to use: 'TabTransformer'")
    parser.add_argument("--num_features", type=int, default=97, help="Number of input features for the tabular model")  
    
    args = parser.parse_args()
    return args
######################### PREPROCESSING FUNCTIONS ################################
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
        path_id_train_1 = os.path.join(files_path, 'train', 'CHD')
        path_id_train_2 = os.path.join(files_path, 'train', 'Non_CHD')
        path_id_test_1 = os.path.join(files_path, 'test', 'CHD')
        path_id_test_2 = os.path.join(files_path, 'test', 'Non_CHD')

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

############TRAINING AND VALIDATION###############################

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id, seed):
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)

# Cargar datos
def load_data(file, seed):
    """
    Loads data from a JSON preprocessed file and returns features and labels that 
    correspond to a training or validation fold.
    Args:
        file (str): Path to the JSON file containing the preprocessed data.
        seed (int): Random seed for reproducibility.
    Returns:
        X (np.ndarray): Features extracted from the JSON file.
        y (np.ndarray): Labels corresponding to the features.
    """
    with open(file, 'r') as file:
        data = json.load(file)
    X, y = [], []
    for i in data:
        embedding = []
        embedding.extend(i['pathological_history_woe'])
        embedding.extend(i['hereditary_history_woe'])
        embedding.extend(i['pharmacological_history_woe'])
        for feature in [
            'platelets', 'hemoglobin', 'hematocrit', 'white_blood_cells', 'neutrophils',
            'age', 'gestational_week_of_imaging', 'body_mass_index', 'gestational_age', 
            'chromosomal_abnormality',
            'screening_procedures', 'thromboembolic_risk',  
            'psychosocial_risk', 'depression_screening',
            'tobacco_use', 'nutritional_deficiencies', 'alcohol_use', 'physical_activity', 
            'pregnancies',
            'vaginal_births', 'cesarean_sections', 'miscarriages', 'ectopic_pregnancies'
        ]:
            value = i.get(feature, -1)
            if isinstance(value, list):
                embedding.extend(value)
            elif isinstance(value, (int, float)):
                embedding.append(value)
        X.append(embedding)
        #y.append(i['cardiopatia'])
        y.append(i['CHD'])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def load_fold(fold_number, output_dir, seed):
    X_train, y_train = load_data(f'{output_dir}/delfos_processed_dataset_fold_{fold_number + 1}_trainf.json', seed)
    X_val, y_val = load_data(f'/{output_dir}/delfos_processed_dataset_fold_{fold_number + 1}_testf.json', seed)
    
    return X_train, y_train, X_val, y_val
