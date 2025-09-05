import json
import os
import numpy as np
import pandas as pd
import sys
from pathlib import Path
#from tabular_script.tab_utils import * 
from utils import *

class Clinical_Record_Preprocessing:
    """
    Preprocesses clinical records for tabular data analysis.
    Args:
        input_file (str): Path to the input JSON file containing clinical records.
        image_folder_path (str): Path to the folder containing images.
        output_dir (str, optional): Directory to save the processed folds in JSON files.
        complete_output_dir (str, optional): Directory to save the complete dataset in a unique JSON file.
    Functions:
        run(): Executes the preprocessing pipeline.
    """
    def __init__(self,
                 input_file,
                 image_folder_path,
                 output_dir=None,
                 complete_output_dir=None):
        self.input_file = input_file
        self.image_folder_path = image_folder_path
        self.output_dir = output_dir
        self.complete_output_dir = complete_output_dir

    def run(self):
        with open(self.input_file, 'r') as file:
            data = json.load(file)

        df = pd.DataFrame(data)

        categorical_features = ["pathological_history", "hereditary_history", "pharmacological_history"]
    
        max_lengths_categorical = [8, 8, 9]

        df_padded = pad_categorical_lists(df, categorical_features, max_lengths_categorical)

        for feature, max_len in zip(categorical_features, max_lengths_categorical):
            df_padded = woe_encoder_list(df_padded, feature, "CHD", max_len)
            
        categorical_features_woe = ["pathological_history_woe", "hereditary_history_woe", "pharmacological_history_woe"]
        df_padded = normalize_categorical_woe(df_padded, categorical_features_woe)

        numeric_features = [
            "platelets", "hemoglobin", "hematocrit", "white_blood_cells",
            "neutrophils", "age", "gestational_week_of_imaging", "body_mass_index", "gestational_age"
        ]
        
        max_lengths_numeric = [2, 2, 2, 2, 10, 10, 10, 10, 10]

        df_padded = df_padded.apply(lambda row: pad_numeric_features(row, numeric_features, max_lengths_numeric), axis=1)

        data_padded = df_padded.to_dict(orient="records")
        standardized_data = standardize_data(data_padded, numeric_features, [
    'chromosomal_abnormality',
    'screening_procedures', 
    'thromboembolic_risk', 'psychosocial_risk', 'depression_screening',
    'tobacco_use', 'nutritional_deficiencies',
    'alcohol_use', 'physical_activity', 'pregnancies', 'vaginal_births',
    'cesarean_sections', 'miscarriages', 'ectopic_pregnancies'
        ])

        if self.complete_output_dir:
            os.makedirs(self.complete_output_dir, exist_ok=True)
            output_file = os.path.join(self.complete_output_dir, "delfos_clinical_data_woe_wnm_standarized_f_normalized.json")
            with open(output_file, 'w') as file:
                json.dump(standardized_data, file, indent=4)

        file_names = ['fold_1', 'fold_2', 'fold_3']
        fold_1_train, fold_1_test, fold_2_train, fold_2_test, fold_3_train, fold_3_test = create_folds(
            standardized_data, self.image_folder_path, file_names
        )

        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            fold_files = {
                "fold_1_train": fold_1_train,
                "fold_1_test": fold_1_test,
                "fold_2_train": fold_2_train,
                "fold_2_test": fold_2_test,
                "fold_3_train": fold_3_train,
                "fold_3_test": fold_3_test
            }

            for fold_name, fold_data in fold_files.items():
                output_file = os.path.join(self.output_dir, f"delfos_processed_dataset_{fold_name}f.json")
                with open(output_file, 'w') as file:
                    json.dump(fold_data, file, indent=4)
