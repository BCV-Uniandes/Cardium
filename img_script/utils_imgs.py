import argparse
import torch
from PIL import Image
from collections import defaultdict, deque
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, precision_recall_curve, roc_auc_score, classification_report
from tqdm import tqdm
import torch
import numpy as np
import random
import matplotlib.pyplot as plt

str2bool = lambda x: (str(x).lower() == 'true')

def get_main_parser():
    parser = argparse.ArgumentParser(description="Training configuration for Vision Transformer")

    # Add arguments
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate for optimizer")
    parser.add_argument("--n_classes", type=int, default=1, help="Number of output classes")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Embedding dimension")
    parser.add_argument("--loss_factor", type=float, default=2, help="factor to multiply weight_loss")
    parser.add_argument("--img_pretrain", type=str2bool, default=True, help="True for pretrained image model")
    parser.add_argument("--img_model", type=str, default="vit_small", help="whether to use medvit, vit_tiny, vity_small, resnet18, resnet50")
    parser.add_argument("--img_feature_dim", type=int, default=384, help="Dimension of image features to be used in multimodal model")
    parser.add_argument("--img_checkpoint", type=str, default="vit_small", help="Path to image model checkpoint")
    parser.add_argument("--img_path_dropout", type=float, default=0.3, help="Path dropout to be used in multimodal models")
    parser.add_argument("--img_class_dropout", type=float, default=0.2, help="Path dropout to be used in multimodal models")
    parser.add_argument("--folds", type=int, default=3, help="Number of folds in k-fold cross validation")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name for optimization threshold")
    parser.add_argument("--sampling", type=str2bool, default=True, help="Whether to use weighted random sampling in dataloader or not")
    parser.add_argument("--smooth_label", type=str2bool, default=False, help="Whether to use smooth label or not")
    parser.add_argument("--trimester", type=str, default=None, help="Which trimester to evaluate")
    parser.add_argument("--data_path", type=str, default=None, help="Image data path for training and evaluation")
    parser.add_argument("--wandb_project", type=str, default="CARDIUM", help="WandB project name")
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
    
    
def plot_prcurve(pr_curve, path):
    # Plot PR curves for train and test
    plt.figure(figsize=(12, 6))

    # Train PR curve subplot
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, first plot
    for i, (precision, recall) in enumerate(zip(pr_curve["precision_train"], pr_curve["recall_train"]), 1):
        plt.plot(recall, precision, label=f'Fold {i}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Train Precision-Recall Curve')
    plt.legend()
    plt.grid()

    # Test PR curve subplot
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, second plot
    for i, (precision, recall) in enumerate(zip(pr_curve["precision_test"], pr_curve["recall_test"]), 1):
        plt.plot(recall, precision, label=f'Fold {i}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Test Precision-Recall Curve')
    plt.legend()
    plt.grid()


    # Save figure
    plt.tight_layout()
    plt.savefig(path)
    
    
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

def get_best_threshold(y_true, y_score, num_thresholds=1000):
    """
    Encuentra el mejor threshold basado en el F1-score generando manualmente los umbrales.

    Args:
        y_true (np.array): Etiquetas verdaderas (0 o 1).
        y_score (np.array): Puntajes o probabilidades del modelo.
        num_thresholds (int): Número de umbrales a probar (default=100).

    Returns:
        best_threshold (float): Mejor threshold basado en el F1-score.
        precision_list (np.array): Lista de valores de precisión.
        recall_list (np.array): Lista de valores de recall.
        thresholds (np.array): Lista de thresholds usados.
    """
    # Generar manualmente los thresholds en el rango [0,1]
    thresholds = np.linspace(0, 1, num_thresholds)

    precision_list = []
    recall_list = []
    f1_scores = []

    for thresh in thresholds:
        y_pred = (y_score >= thresh).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=1)
        recall = recall_score(y_true, y_pred, zero_division=1)
        f1 = f1_score(y_true, y_pred, zero_division=1)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_scores.append(f1)

    # Encontrar el índice del mejor F1-score
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    return best_threshold, np.array(precision_list), np.array(recall_list), np.array(f1_scores), thresholds


def evaluate_threshold_img(model, loader, device, threshold=0.5):
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
            img_data, patient_ids = inputs[0].to(device), inputs[1]
            targets = targets.unsqueeze(1).to(device)

            # Forward pass
            outputs = model(img_data)
            targets = targets.to(torch.float32)

            # Sigmoid to convert logits to probabilities
            probabilities = torch.sigmoid(outputs).cpu().numpy()
            targets = targets.cpu().numpy()

            # Group predictions and true labels by patient_id
            for patient_id, prob, target in zip(patient_ids, probabilities, targets):
                if patient_id not in y_score_patient:
                    y_score_patient[patient_id] = np.array([prob])  # Initialize as array
                else:
                    y_score_patient[patient_id] = np.append(y_score_patient[patient_id], prob)  # Append to array

                y_true_patient[patient_id] = target 

    # Average scores for each patient
    y_score_patient_avg = {pid: sum(scores) / len(scores) for pid, scores in y_score_patient.items()}
    y_true_patient = {pid: int(label) for pid, label in y_true_patient.items()}

    # Convert to arrays for metric computation
    y_true = list(y_true_patient.values())
    y_score = list(y_score_patient_avg.values())
    y_pred = [1 if score > threshold else 0 for score in y_score]  # Apply threshold of 0.5

    # Calculate metrics
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["No Cardiopatia", "Cardiopatia"], output_dict=True)

    return precision, recall, accuracy, f1, roc_auc, report, y_score, y_true

def find_best_threshold_img(model, loader, device):
    """
    Find the best threshold based on the F1-score using the validation set.

    Args:
        model (torch.nn.Module): The model to evaluate.
        loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        device (torch.device): Device to perform computations on (e.g., 'cuda' or 'cpu').

    Returns:
        float: Best threshold for classification based on F1-score.
    """
    y_true_patient = {}
    y_score_patient = defaultdict(list)

    model.eval()
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Finding Best Threshold"):
            img_data, patient_ids = inputs[0].to(device), inputs[1]
            targets = targets.to(device)
            
            outputs = model(img_data)
            outputs = outputs.squeeze(1)
            
            probabilities = torch.sigmoid(outputs).cpu().numpy()
            targets = targets.cpu().numpy()
            
            # Group predictions and true labels by patient_id
            for patient_id, prob, target in zip(patient_ids, probabilities, targets):
                if patient_id not in y_score_patient:
                    y_score_patient[patient_id] = np.array([prob])  # Initialize as array
                else:
                    y_score_patient[patient_id] = np.append(y_score_patient[patient_id], prob)  # Append to array

                y_true_patient[patient_id] = target 

    # Average scores for each patient
    y_score_patient_avg = {pid: sum(scores) / len(scores) for pid, scores in y_score_patient.items()}
    y_true_patient = {pid: int(label) for pid, label in y_true_patient.items()}

    # Convert to arrays for metric computation
    y_true = list(y_true_patient.values())
    y_score = list(y_score_patient_avg.values())

    # Get best threshold
    best_threshold, precision, recall, f1_score, thresholds = get_best_threshold(y_true, y_score)

    return best_threshold, precision, recall, f1_score, thresholds

#TODO: clean utils (eliminate functions not beign used)
def prepare_images(images, labels, general_transform, augmented_tranforms):

    X_augmented = []
    y_augmented = []
    
    for img_path, label in zip(images, labels):
        # Open the image
        #breakpoint()
        image = Image.open(img_path).convert("RGB")
        
        # Conditional augmentation
        if label == 1:  # Example: Flip for label 1
            for i in range(len(augmented_tranforms)):
                augmented_image = augmented_tranforms[i](image)
                X_augmented.append(augmented_image)
                y_augmented.append(label)
        
        image = general_transform(image)
        # Append to the augmented dataset
        X_augmented.append(image)
        y_augmented.append(label)
    
    return X_augmented, y_augmented