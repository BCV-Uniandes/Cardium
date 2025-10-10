import os
from PIL import Image
import numpy as np

dataset_path = '/home/dvegaa/Cardium/dataset/cardium_images'  # Ruta de tu dataset
new_dataset_path = '/home/dvegaa/Cardium/dataset/cardium_new_images'  # Nueva ruta para guardar las imágenes con negro

# Asegúrate de que el directorio destino existe
os.makedirs(new_dataset_path, exist_ok=True)

folds = os.listdir(dataset_path)
for fold in folds:
    fold_path = os.path.join(dataset_path, fold)
    for subset in os.listdir(fold_path):
        subset_path = os.path.join(fold_path, subset)
        for lab in os.listdir(subset_path):
            lab_path = os.path.join(subset_path, lab)
            for patient in os.listdir(lab_path):
                patient_path = os.path.join(lab_path, patient)

                # Crear la misma estructura de directorios en la carpeta de destino
                relative_path = os.path.relpath(patient_path, dataset_path)
                new_patient_path = os.path.join(new_dataset_path, relative_path)
                os.makedirs(new_patient_path, exist_ok=True)

