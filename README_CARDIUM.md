# CARDIUM Dataset and Pretrained Models

This Google Drive contains the **CARDIUM dataset** together with the **pretrained model weights** used in our paper for congenital heart disease (CHD) detection from prenatal ultrasound images and clinical (tabular) data.
---

## 1. Dataset Overview

The CARDIUM dataset is composed of:

* **Ultrasound images** collected from multiple patients
* **A single set of clinical (tabular) data per patient**

Each patient may have **multiple ultrasound images**, acquired:

* at different gestational ages,
* on different dates,
* from different imaging planes and viewing angles.

In contrast, the **clinical data are unique per patient** and shared across all images belonging to that patient.

---

## 2. Image Dataset and Cross-Validation Folds

All ultrasound images used in the paper are stored in the following compressed archive:

```
cardium_images.tar.gz
```

After extraction, this archive contains the dataset organized into **three folds**, corresponding to the **3-fold cross-validation** strategy used in our experiments.

```
cardium_images/
 ├── fold_1/
 ├── fold_2/
 └── fold_3/
```

Each fold contains a **train** and **test** split:

```
fold_k/
 ├── train/
 └── test/
```

Within each split, images are organized by diagnostic category:

```
train/
 ├── CHD/
 └── Non_CHD/
```

* **CHD**: Images from patients diagnosed with congenital heart disease
* **Non_CHD**: Images from patients without the condition

Inside each category folder, images are grouped by **patient ID**:

```
CHD/
 ├── Patient_001/
 ├── Patient_002/
 └── ...
```

Each patient folder contains all ultrasound images corresponding to that patient.

This structure ensures that **no patient appears in both the training and test splits within the same fold**.

---

## 3. Full Dataset (Without Predefined Splits)

To facilitate use of the dataset beyond the predefined experimental splits, we additionally provide:

```
CARDIUM_dataset/
```

This folder contains the **entire image dataset**, organized only by diagnostic category and patient ID, without train/test separation:

```
CARDIUM_dataset/
 ├── CHD/
 │    ├── Patient_001/
 │    └── ...
 └── Non_CHD/
      ├── Patient_101/
      └── ...
```

This structure allows users to:

* Create custom train/test splits
* Perform patient-level cross-validation
* Use the full dataset independently of our experimental protocol

---

## 4. Clinical Labels and Tabular Data

Two JSON files provide the clinical information associated with each patient.

### 4.1 Raw clinical data

**`cardium_clinical_data_wnm_translated_final_cleaned.json`**

* Dictionary indexed by **patient ID**
* Patient IDs match the folder names used in the image dataset
* Values contain the **raw clinical and demographic variables**
* One clinical record per patient

Example structure:

```json
{
  "Patient_001": {
    "feature_1": value,
    "feature_2": value
  }
}
```

---

### 4.2 Preprocessed clinical data

**`cardium_clinical_data_woe_standardized_f_normalized.json`**

* Same structure and indexing as the raw file
* Contains the clinical data after preprocessing, including:

  * Cleaning
  * Encoding
  * Standardization
  * Normalization

The full preprocessing pipeline is described in **Section 3.5** of our paper:
[https://arxiv.org/pdf/2510.15208](https://arxiv.org/pdf/2510.15208)

---

## 5. Pretrained Models

The pretrained weights used in the paper are provided as three compressed archives:

```
image_encoder.tar
tabular_encoder.tar
cardium_model_weights.tar
```

Each archive contains checkpoints for the **three cross-validation folds**.

---

### 5.1 Image encoder

**`image_encoder.tar`**

After extraction:

```
image_encoder/
 ├── fold0_best_model.pth
 ├── fold1_best_model.pth
 └── fold2_best_model.pth
```

These checkpoints correspond to the image encoder trained on the training split of each fold.

---

### 5.2 Tabular encoder

**`tabular_encoder.tar`**

After extraction:

```
tabular_encoder/
 ├── fold0_best_model.pth
 ├── fold1_best_model.pth
 └── fold2_best_model.pth
```

These checkpoints correspond to the tabular encoder trained on the preprocessed clinical data for each fold.

---

### 5.3 Multimodal CARDIUM model

**`cardium_model_weights.tar`**

After extraction:

```
cardium_model_weights/
 ├── fold0_best_model.pth
 ├── fold1_best_model.pth
 └── fold2_best_model.pth
```

Each checkpoint corresponds to the **multimodal CARDIUM model**, trained using only the training data of its respective fold and evaluated on the held-out test split.

---

## 6. Reference

If you use this dataset or the provided pretrained models, please cite:

**CARDIUM: Multimodal Learning for Prenatal Congenital Heart Disease Detection**
[https://arxiv.org/pdf/2510.15208](https://arxiv.org/pdf/2510.15208)

---

## 7. Contact

We are continuously improving the dataset organization and documentation.
If you do not see any of the files described above, or if you have further questions, please feel free to reach out.