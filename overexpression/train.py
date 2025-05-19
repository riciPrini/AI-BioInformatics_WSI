import os
import torch
import numpy as np
import random
from tqdm import tqdm
import pandas as pd
import argparse
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.optim import Adam, RAdam, AdamW
import torch.nn as nn
from models import *
from dataset import UNIDataset
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser(description='BRCA overexprection')
parser.add_argument('--seed', '-s', type=int, default=17,
                        help='Random seed')
parser.add_argument('--data_directory', type=str, default='/work/ai4bio2024/brca_surv/LAB_WSI_Genomics/TCGA_BRCA/Data/wsi/features_UNI/pt_files',
                        help='Dataset directory')
parser.add_argument('--labels_file', type=str, default='/work/ai4bio2024/brca_surv/dataset/dataset_brca_wsi.csv',
                        help='label file path')
parser.add_argument('--label', type=str, default='BRCA1',
                        help='Label to use for training')
parser.add_argument('--epoch', '-e', type=int, default=10,
                    help='Number of epochs')
parser.add_argument('--folds', '-f', type=int, default=5,
                    help='Number of folds')

args = parser.parse_args()

SEED = args.seed
EPOCHS = args.epoch
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

print(f'Predicting {args.label} overexprection')


#####################################################################à

class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.save_weights = True

    def __call__(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
            self.save_weights = True
            if self.verbose:
                print(f'New best loss: {self.best_loss:.4f}')
        else:
            self.save_weights = False
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                return True
        return False



csv_file = args.labels_file
data_dir = args.data_directory

data_frame = pd.read_csv(csv_file)


train_val_df, test_df = train_test_split(data_frame, test_size=0.2, stratify = data_frame[args.label], random_state=SEED)
y = train_val_df[args.label]
X = train_val_df.drop(args.label, axis = 1)

skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

loss_train_list = []
loss_val_list = []

plt.figure(figsize=(10, 15))

for f, (train_index, val_index) in enumerate(skf.split(X, y)):
    print(f'\n#########  Fold {f+1}/{FOLDS}  #########\n')

    train_dataset = UNIDataset(data_frame=train_val_df.iloc[train_index], data_dir=data_dir, label = args.label, seed=SEED, max_patches=4096)
    val_dataset = UNIDataset(data_frame=train_val_df.iloc[val_index], data_dir=data_dir, label = args.label, seed=SEED, max_patches=4096)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=1)

    LR = 0.0001
    WEIGHT_DECAY = 0.001
    NUM_ACCUMULATION_STEPS = 8
    PATIENCE = 5

    model = ABMIL().to(device)
    #optimizer = RAdam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCELoss().to(device)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True)

    loss_train = []
    loss_val = []


    for e in range(EPOCHS):
        model.train()
        running_loss= 0.0

        for idx, (data, label) in enumerate(tqdm(train_loader)):
            data = data.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            output, _ = model(data)
            loss = criterion(output, label)

            running_loss += loss.item()
            loss.backward()

            optimizer.step()

        scheduler.step()
        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch {e + 1}/{EPOCHS}, Training Loss: {epoch_loss:.4f}')
        loss_train.append(epoch_loss)

        torch.cuda.empty_cache()

        #Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, label in tqdm(val_loader):
                data = data.to(device)
                label = label.to(device)
                output, _ = model(data)
                loss = criterion(output, label)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f'Epoch {e+1}/{EPOCHS}, Validation Loss: {val_loss:.4f}')
        loss_val.append(val_loss)


        if early_stopping(val_loss):
            print("Early stopped")
            break

        if early_stopping.save_weights:
            torch.save(model.state_dict(), f'./model_weights_{f+1}.pth')

    loss_train_list.append(loss_train)
    loss_val_list.append(loss_val)

#####  Plot  #####

plt.figure(figsize=(10, 5*FOLDS))

for i in range(len(loss_train_list)):
    loss_train = loss_train_list[i]
    loss_val = loss_val_list[i]
    epochs = range(1, len(loss_train) + 1)

    loss_train = [round(loss, 4) for loss in loss_train]
    loss_val = [round(loss, 4) for loss in loss_val]

    plt.subplot(len(loss_train_list), 1, i + 1)  # (righe, colonne, indice)
    plt.plot(epochs, loss_train, label='Training Loss', marker='o', linestyle='-', color='red')
    plt.plot(epochs, loss_val, label='Validation Loss', marker='o', linestyle='-', color='blue')

    plt.title(f'Fold {i+1}/{FOLDS}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(epochs)
    plt.legend()
    plt.grid()


plt.tight_layout()
plt.savefig('./loss_plot.png')
#plt.show()

print('Done!')
