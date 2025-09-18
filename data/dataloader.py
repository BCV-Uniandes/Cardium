import os
from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
import os
import os.path
from PIL import Image
import random
import json

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if using multi-GPU
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class CardiumDataset(Dataset):
    def __init__(self, root, transform, json_root="", multimodal=False):
        self.classes = ["CHD", "Non_CHD"]
        self.root = root
        self.json_root = json_root
        self.transform = transform
        self.multimodal = multimodal

        self.patient_info = {}
        self.data = []
        self.views = []
        self.labels = []
        self.cache = {}
        self.class_to_idx = {'CHD': 1, 'Non_CHD': 0}
        
        if multimodal:
            self._load_patient_info()
        # Load images and labels
        self._load_data()

    def _load_patient_info(self):
        """Load patient information from JSON."""
        if not os.path.exists(self.json_root):
            raise FileNotFoundError(f"JSON file not found: {self.json_root}")

        with open(self.json_root, "r") as f:
            data = json.load(f)

        for patient in data:
            patient_id = patient['id']
            embedding = []

            # Ensure features are extracted correctly (like in load_data())
            embedding.extend(patient.get('pathological_history_woe', []))
            embedding.extend(patient.get('hereditary_history_woe', []))
            embedding.extend(patient.get('pharmacological_history_woe', []))

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
                value = patient.get(feature, -1)  # Default to -1 if feature is missing
                if isinstance(value, list):
                    embedding.extend(value)  # Ensure all lists are fully extended
                elif isinstance(value, (int, float)):
                    embedding.append(value)

            # Store extracted features
            self.patient_info[patient_id] = np.array(embedding, dtype=np.float32)

            # Debugging: Print a sample patient feature vector
            if len(self.patient_info) == 1:
                print(f"Sample patient features (DelfosDataset): {len(embedding)} features")

    def _load_data(self):
        for class_name in self.classes:
            class_folder = os.path.join(self.root, class_name)
            images_dict = {}
            if not os.path.exists(class_folder):
                print(f"Warning: Class folder not found: {class_folder}")
                continue
            
            # Obtener y ordenar los directorios
            folders = sorted(os.listdir(class_folder))
            #folders = os.listdir(class_folder)
            for id_folder in folders:
                id_path = os.path.join(class_folder, id_folder)
                if not os.path.isdir(id_path):
                    continue

                if self.multimodal:
                    # Check if patient info exists for this ID
                    patient_info = self.patient_info.get(id_folder)
                    if patient_info is None:
                        print(f"Warning: No patient info found for {id_folder}")
                        continue

                images = sorted(os.listdir(id_path), key=lambda x: int(x.split('_')[0]))
                #images_dict[id_folder] = images
                for image in images:
                    image_path = os.path.join(id_path, image)
                    if image.lower().endswith((".png", ".jpg", ".jpeg")):
                        image_path = os.path.join(id_path, image)
                        label = self.class_to_idx[class_name]
                        self.labels.append(label)
                        if self.multimodal:
                            self.data.append((image_path, patient_info, id_folder, label))
                        else:
                            self.data.append((image_path, id_folder, label))

            #name = class_folder.split("/")
            #with open(f"/home/dvegaa/Cardium/img_script/jsons_images_anon/{name[-3]}_{name[-2]}_{name[-1]}.json", "w") as f:
            #    json.dump(images_dict, f, indent=4)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.multimodal:
            image_path, patient_info, patient_id, label = self.data[idx]
        else:
            image_path, patient_id, label = self.data[idx]

        if image_path in self.cache:
            image = self.cache[image_path]
        else:
            image = Image.open(image_path).convert("RGB")
            image = self.transform(np.array(image))
            self.cache[image_path] = image
        
        if self.multimodal:
            return (image, torch.tensor(patient_info, dtype=torch.float32), patient_id), label 
        else: 
            return (image, patient_id), label