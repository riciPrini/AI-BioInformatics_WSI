import os
import torch
import numpy as np
import random
from tqdm import tqdm
import pandas as pd
import argparse
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from models import *
from dataset import UNIDataset
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    log_loss,
    accuracy_score,
    matthews_corrcoef,
    classification_report
)

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

parser = argparse.ArgumentParser(description='BRCA overexprection')
parser.add_argument('--seed', '-s', type=int, default=17,
                        help='Random seed')#42
parser.add_argument('--data_directory', type=str, default='/work/ai4bio2024/brca_surv/LAB_WSI_Genomics/TCGA_BRCA/Data/wsi/features_UNI/pt_files',
                        help='Dataset directory')
parser.add_argument('--labels_file', type=str, default='/homes/fdoronzio/ai4bio/data/dataset_brca_wsi_regression.csv',
                        help='label file path')
parser.add_argument('--label', type=str, default='BRCA1',
                        help='Label to use for training')
parser.add_argument('--folds', '-f', type=int, default=5,
                    help='Number of folds')

args = parser.parse_args()

SEED = args.seed
FOLDS = args.folds

print(f'Seed: {SEED}')

def setup(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # If using CUDA.
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


setup(SEED)

if torch.cuda.is_available():
    device = 'cuda'
    print(f'# CUDA available: {torch.cuda.get_device_name(0)}')
else:
    device = 'cpu'
    print('No GPU!')

print(f'Testing {args.label} over exprection\n')


csv_file = args.labels_file
data_dir = args.data_directory

data_frame = pd.read_csv(csv_file)

_ , test_df = train_test_split(data_frame, test_size=0.2, random_state=SEED)

test_dataset = UNIDataset(data_frame=test_df, data_dir=data_dir, label = args.label, seed=SEED, max_patches=4096)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=1)

results = {
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1': [],
    'mcc': [],
    'confusion_matrices': [],

    'mae' : [],
    'mse' : [],
    'rmse' : [],
    'r2' : []
}

###############################################
def mean_absolute_error(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred)).item()


def mean_squared_error(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2).item()


def root_mean_squared_error(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))


def r_squared(y_true, y_pred):
    ss_total = torch.sum((y_true - torch.mean(y_true)) ** 2)
    ss_residual = torch.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total).item()

################################################

for f in range(FOLDS):

    print(f'######## Testing model {f+1} ########\n')

    model = ABMIL(use_layernorm=True)
    #model = DS_ABMIL()


    model.load_state_dict(torch.load(f'./model_weights_{f+1}.pth'))
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        y_pred = []
        y_true = []
        for data, label in tqdm(test_loader):
            data = data.to(device)
            label = label.to(device)
            output, _ = model(data)
            y_pred.append(output)
            y_true.append(label)

        y_pred = torch.cat(y_pred)
        y_true = torch.cat(y_true)

    y_pred = y_pred.cpu()
    y_true = y_true.cpu()


    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r_squared(y_true, y_pred)
    
    results['mae'].append(mae)
    results['mse'].append(mse)
    results['rmse'].append(rmse)
    results['r2'].append(r2)

    print(f'Test MAE: {mae:.4f}')
    print(f'Test MSE: {mse:.4f}')
    print(f'Test RMSE: {rmse:.4f}')
    print(f'Test R²: {r2:.4f}')
    
    # Fai parte classification as regression
    # prendi soglia a mano...

    threshold = 4.7924171717171715

    y_pred = np.where(y_pred.numpy() > threshold, 1, 0)
    y_true = np.where(y_true.numpy() > threshold, 1, 0)

    
    #y_pred_binary = (y_pred > 0.5).float()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred, normalize='true')
    mcc = matthews_corrcoef(y_true, y_pred)
    class_report = classification_report(y_true, y_pred)

    results['accuracy'].append(accuracy)
    results['precision'].append(precision)
    results['recall'].append(recall)
    results['f1'].append(f1)
    results['mcc'].append(mcc)
    results['confusion_matrices'].append(conf_matrix)

    print(f"Accuracy: {accuracy:.4f}\n")
    print(f"Precision: {precision:.4f}\n")
    print(f"Recall: {recall:.4f}\n")
    print(f"F1 Score: {f1:.4f}\n")
    print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}\n")
    print(f"Confusion Matrix:\n{np.round(conf_matrix, 2)}\n")
    print(f"Classification Report:\n{class_report}\n")

mean_results = {metric: np.mean(values) for metric, values in results.items() if metric not in ['confusion_matrices']}
std_results = {metric: np.std(values) for metric, values in results.items() if metric not in ['confusion_matrices']}
cf_mean = np.mean(results['confusion_matrices'], axis=0)
cf_std = np.std(results['confusion_matrices'], axis=0)

print("########################################\n")
print("############# Final Report #############\n")
print("########################################\n")
for metric, mean_value in mean_results.items():
    std_value = std_results[metric]
    print(f"{metric.capitalize()}: {mean_value:.4f} ± {std_value:.4f}\n")
print(f'Confusion Matrix:\n{np.round(cf_mean, 2)}')
print(f'Confusion Matrix std:\n{np.round(cf_std, 2)}')

metric = 'accuracy'
best_model_index = np.argmax(results[metric])
print(f"\nBest model for {metric}: {best_model_index + 1} with {results[metric][best_model_index]:.4f}")

metric = 'f1'
best_model_index = np.argmax(results[metric])
print(f"\nBest model for {metric}: {best_model_index + 1} with {results[metric][best_model_index]:.4f}")






