import os
from PIL import Image
import numpy as np

dataset_path = '/home/dvegaa/Cardium/dataset/cardium_images'  # Ruta de tu dataset
new_dataset_path = '/home/dvegaa/Cardium/dataset/black_cardium_images'  # Nueva ruta para guardar las imágenes con negro

CROP_TOP = 60  # Cubrir las primeras 60 filas con negro

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
                for img_name in os.listdir(patient_path):
                    img_path = os.path.join(patient_path, img_name)
                    
                    # Abrir la imagen con Pillow (PIL)
                    try:
                        img = Image.open(img_path)
                    except IOError:
                        continue  # Si no se puede abrir la imagen, la ignoramos

                    # Convertir la imagen a un array de NumPy para manipulación
                    img_array = np.array(img)

                    # Cubrir las primeras 60 filas con negro (0)
                    img_array[:CROP_TOP, :, :] = 0  # Establecer las primeras 60 filas a negro (0)

                    # Convertir de nuevo el array a imagen de PIL
                    img_modified = Image.fromarray(img_array)

                    # Crear la misma estructura de directorios en la carpeta de destino
                    relative_path = os.path.relpath(patient_path, dataset_path)
                    new_patient_path = os.path.join(new_dataset_path, relative_path)
                    os.makedirs(new_patient_path, exist_ok=True)

                    # Guardar la imagen modificada en la misma ruta dentro de 'new_dataset_path'
                    # Guardar como PNG (sin compresión con pérdida)
                    #breakpoint()
                    img_name = img_name.split(".")[0]
                    img_modified.save(os.path.join(new_patient_path, f"{img_name}.png"), format='PNG')

print(f"Imágenes con las primeras {CROP_TOP} filas cubiertas de negro guardadas en: {new_dataset_path}")