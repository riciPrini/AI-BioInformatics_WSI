# ========== Standard Library ==========
import os
import random
import argparse
import warnings
from copy import deepcopy

# ========== Third-Party Libraries ==========
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from munch import Munch
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import Adam, RAdam
from imblearn.over_sampling import SMOTE

# ========== Local Modules ==========
from models import *
from models.model_coattn import MCAT_Surv, MCAT_Surv_SingleOmic
from models.dataset_survival import *
from dataset import UNIDataset
from utils.core_utils import *
from utils.utils import *

# ========== Suppress Warnings ==========
warnings.filterwarnings("ignore", category=UserWarning)

# ========== Argument Parser ==========
parser = argparse.ArgumentParser(description='BRCA Overexpression')

parser.add_argument('--seed', '-s', type=int, default=42, help='Random seed')
parser.add_argument('--data_directory', type=str, default='/work/ai4bio2024/brca_surv/LAB_WSI_Genomics/TCGA_BRCA/Data/wsi/features_UNI/pt_files', help='Dataset directory')
parser.add_argument('--labels_file', type=str, default='/work/ai4bio2024/brca_surv/dataset/genomics_csv/dataset_survival_fpkm_unstranded.csv', help='Label file path')
parser.add_argument('--apply_sig', action='store_true', default=False, help='Use genomic features as signature embeddings.')
parser.add_argument('--label', type=str, default='BRCA2', help='Label to use for training')
parser.add_argument('--epoch', '-e', type=int, default=10, help='Number of epochs')
parser.add_argument('--folds', '-f', type=int, default=5, help='Number of folds')
parser.add_argument('--reg_type', type=str, choices=['None', 'omic', 'pathomic'], default='None', help='L1-Regularization module (default: None)')
parser.add_argument('--mode', type=str, choices=['omic', 'path', 'pathomic', 'cluster', 'coattn'], default='coattn', help='Modality to use')
parser.add_argument('--split_dir', type=str, default='/work/ai4bio2024/brca_surv/dataset/split/', help='Split directory')

args = parser.parse_args()

# ========== Global Settings ==========
SEED = args.seed
EPOCHS = args.epoch
FOLDS = args.folds
MODEL = "MCAT"

print(f'Predicting Overall Survival - {args.label}_{EPOCHS}')

# ========== Seed Setup ==========
def setup(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

setup(SEED)

# ========== Paths ==========
results_dir = "/work/ai4bio2024/brca_surv/survival/MCAT_genomics/checkpoint/fpkm_unstranded/"
csv_file = args.labels_file
data_dir = args.data_directory

# ========== Dataset Configuration ==========
TCGA_BRCA_dataset_config = {
    "name": "TCGA_BRCA",
    "parameters": {
        "dataframe_path": csv_file,
        "pt_files_path": data_dir,
        "genomics_path": "/work/ai4bio2024/brca_surv/LAB_WSI_Genomics/TCGA_BRCA/Data/genomic/fpkm_unstranded.csv", # fpkm_uq_unstranded # fpkm_unstranded # tpm_unstranded # unstranded
        "tissue_type_filter": [],
        "label_name": "FUT",
        "censorships_name": "Survival",
        "case_id_name": "case_id",
        "slide_id_name": "slide_id",
    }
}
TCGA_BRCA_dataset_config = Munch.fromDict(TCGA_BRCA_dataset_config)
#####################################################################à

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, warmup=5, patience=15, stop_epoch=10, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.warmup = warmup
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss
        # print(self.counter)
        if epoch < self.warmup:
            pass
        elif self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}\n')
            #print(f'EarlyStopping counter: {self.counter} out of {self.patience}, epoch:{epoch}, stop_epoch {self.stop_epoch}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            # self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss




class Multimodal_WSI_Genomic_Dataset(Dataset):
    def __init__(self,  datasets_configs = [TCGA_BRCA_dataset_config],
                        task_type="Survival", # Survival or treatment_response
                        max_patches=4096,
                        n_bins=4,
                        label = "BRCA1",
                        eps=1e-6,
                        sample=True,
                        load_slides_in_RAM=False,
                        ):
        self.task_type = task_type
        self.label = label
        self.load_slides_in_RAM = load_slides_in_RAM
        if self.load_slides_in_RAM:
            self.slides_cache = {}
            
        self.datasets = {}
        for i, dataset_config in enumerate(datasets_configs):
            config = dataset_config
            if config.name in self.datasets:
                raise ValueError("Dataset name {} already exists".format(config.name))
            self.datasets[config.name] = config.parameters # asser config.name in datasets
            dataframe = pd.read_csv(config.parameters.dataframe_path,dtype={'case_id': str})
            dataframe = dataframe.dropna()
            dataframe["dataset_name"] = [config.name for _ in range(len(dataframe))]
            if task_type == "Survival":
                rename_dict = { self.datasets[config.name].label_name: "time",
                                self.datasets[config.name].censorships_name: "censorship",
                                self.datasets[config.name].case_id_name: "case_id",
                                self.datasets[config.name].slide_id_name: "slide_id",
                                
                                
                                } 
                dataframe.rename(columns=rename_dict, inplace=True)
                dataframe["time"] = dataframe["time"].astype(int)
                self.case_id_name = "case_id"
                self.slide_id_name = "slide_id"
            else:
                self.case_id_name = self.datasets[config.name].case_id_name
                self.slide_id_name = self.datasets[config.name].slide_id_name
            # dataframe = self.filter_by_tissue_type(config.name, dataframe, config.parameters.tissue_type_filter)

            # load genomics data
            genomics = pd.read_csv(config.parameters.genomics_path, sep="\t", dtype={'Unnamed: 0': str})
            genomics = genomics.set_index("Unnamed: 0").dropna()
            ## Getting BRCA1 and BRCA2
            genomics = genomics[['ENSG00000000003.15', 'ENSG00000000005.6']]


            genomics = np.log(genomics+0.1)
            # print(genomics)
            if i==0:
                self.dataframe = dataframe
                self.genomics = genomics
            else:
                self.dataframe = pd.concat([self.dataframe, dataframe], ignore_index=True)
                self.genomics = pd.concat([self.genomics, genomics], ignore_index=True)
                       
        #{'pAdnL', 'pOvaR', 'pMes1', 'pOth', 'pTubL', 'pPer', 'pAdnR', 'pTubL1', 'pOva', 'pTubR', 'p2Ome2', 'pPer2', 'pVag', 'pLNR', 'pUte1', 
        # 'pPerR1', 'pOvaL1', 'pOvaL', 'p2Oth', 'pPer ', 'pTub', 'pOme2', 'p0Ome', 'pUte2', 'pOva2', 'pMes', 'pOme ', 'pBow', 'pOme1', 'pOth2', 
        # 'pAdnR1', 'pOth1', 'p2Ome1', 'pOme', 'p2Per1', 'pPer3', 'pOvaR1', 'pPerL ', 'pUte', 'pOme3', 'pAndL', 'pTub2', 'pPer1'}
        # self.pt_files_path = pt_files_path
        self.max_patches = max_patches
        self.sample = sample
        self.n_bins = n_bins
        # self.label_name = label_name
        # self.censorships_name = censorships_name
        self.eps = eps
        # self._filter_by_tissue_type()
        self._compute_patient_dict()
        self._compute_patient_df()
        if self.task_type == "Survival":
            self._compute_labels()
        else:
            self.patient_df["label"] = self.patient_df["Treatment_Response"]
        print("Dataset loaded with {} slides and {} patients".format(len(self.dataframe), len(self.patient_df)))


    def _compute_patient_dict(self):
        self.patient_list = list(self.dataframe[self.case_id_name].unique())
        self.patient_dict = {patient: list(self.dataframe[self.dataframe[self.case_id_name] == patient][self.slide_id_name]) for patient in self.patient_list}

    def _compute_patient_df(self):
        self.patient_df = self.dataframe.drop_duplicates(subset=self.case_id_name)
        self.patient_df = self.patient_df.reset_index(drop=True)    
        self.patient_df = self.patient_df.set_index(self.case_id_name, drop=False)

    def get_train_test_val_splits(self, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
        np.random.seed(random_state)
        patients = np.array(self.patient_list)
        np.random.shuffle(patients)
        n = len(patients)
        train_end = int(n * train_size)
        val_end = int(n * (train_size + val_size))
        train_patients = patients[:train_end]
        val_patients = patients[train_end:val_end]
        test_patients = patients[val_end:]

        train_patients_idx = pd.Index(train_patients)
        val_patients_idx = pd.Index(val_patients)
        test_patients_idx = pd.Index(test_patients)
        # for name in self.genomics.columns.tolist():
        #     scaler  = StandardScaler()
        #     self.genomics.loc[train_patients_idx, name] = scaler.fit_transform(self.genomics.loc[train_patients_idx, name].values.reshape(-1,1)).ravel()
        #     self.genomics.loc[val_patients_idx, name]   = scaler.transform(self.genomics.loc[val_patients_idx, name].values.reshape(-1,1)).ravel()
        #     self.genomics.loc[test_patients_idx, name]  = scaler.transform(self.genomics.loc[test_patients_idx, name].values.reshape(-1,1)).ravel()

        self.normalized_genomics = deepcopy(self.genomics)
        X_train = self.normalized_genomics.loc[train_patients_idx, :]
        X_val = self.normalized_genomics.loc[val_patients_idx, :]
        X_test = self.normalized_genomics.loc[test_patients_idx, :]

        scaler = StandardScaler()
        scaler.fit(X_train)  # fit on train set

        # Transform entire subsets of the copied DataFrame
        self.normalized_genomics.loc[train_patients_idx, :] = scaler.transform(X_train)
        self.normalized_genomics.loc[val_patients_idx, :] = scaler.transform(X_val)
        self.normalized_genomics.loc[test_patients_idx, :] = scaler.transform(X_test)

        # train_indices = [i for i, patient in enumerate(self.patient_list) if patient in train_patients]
        # val_indices = [i for i, patient in enumerate(self.patient_list) if patient in val_patients]
        # test_indices = [i for i, patient in enumerate(self.patient_list) if patient in test_patients]
        print("Train: {}, Val: {}, Test: {}".format(len(train_patients), len(val_patients), len(test_patients)))
        assert len(train_patients) + len(val_patients) + len(test_patients) == len(self.patient_list)
        return train_patients, val_patients, test_patients
    

    def _compute_labels(self):
        uncensored_df = self.patient_df[self.patient_df["censorship"] == 0]
        disc_labels, q_bins = pd.qcut(uncensored_df["time"], q=self.n_bins, retbins=True, labels=False, duplicates='drop')
        q_bins[-1] = self.patient_df["time"].max() + self.eps
        q_bins[0] = self.patient_df["time"].min() - self.eps
        
        # assign patients to different bins according to their months' quantiles (on all data)
        # cut will choose bins so that the values of bins are evenly spaced. Each bin may have different frequncies
        disc_labels, q_bins = pd.cut(self.patient_df["time"], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
        self.patient_df.insert(2, 'label', disc_labels.values.astype(int))
        self.bins = q_bins

    def _load_wsi_embs_from_path(self, dataset_name, slide_names):
            """
            Load all the patch embeddings from a list a slide IDs. 

            Args:
                - self 
                - slide_names : List
            
            Returns:
                - patch_features : torch.Tensor 
                - mask : torch.Tensor

            """
            patch_features = []
            pt_files_path = self.datasets[dataset_name].pt_files_path
            # load all slide_names corresponding for the patient
            for slide_id in slide_names:
                if self.load_slides_in_RAM:
                    if slide_id in self.slides_cache:
                        wsi_bag = self.slides_cache[slide_id]
                    else:
                        wsi_path = os.path.join(pt_files_path, '{}.pt'.format(slide_id))
                        wsi_bag = torch.load(wsi_path, weights_only=True)
                        self.slides_cache[slide_id] = wsi_bag
                else:
                    wsi_path = os.path.join(pt_files_path, '{}.pt'.format(slide_id))
                    wsi_bag = torch.load(wsi_path, weights_only=True) # changed to True due to python warning
                patch_features.append(wsi_bag)
            patch_features = torch.cat(patch_features, dim=0)
            # print("patch_features.shape[0]: ", patch_features.shape[0])

            if self.sample:
                max_patches = self.max_patches

                n_samples = min(patch_features.shape[0], max_patches)
                idx = np.sort(np.random.choice(patch_features.shape[0], n_samples, replace=False))
                patch_features = patch_features[idx, :]
                
            
                # make a mask 
                if n_samples == max_patches:
                    # sampled the max num patches, so keep all of them
                    mask = torch.zeros([max_patches])
                else:
                    # sampled fewer than max, so zero pad and add mask
                    original = patch_features.shape[0]
                    how_many_to_add = max_patches - original
                    zeros = torch.zeros([how_many_to_add, patch_features.shape[1]])
                    patch_features = torch.concat([patch_features, zeros], dim=0)
                    mask = torch.concat([torch.zeros([original]), torch.ones([how_many_to_add])])
            
            else:
                mask = torch.zeros([patch_features.shape[0]])

            return patch_features, mask
    
    def set_sample(self, sample):
        self.sample = sample

    def __getitem__(self, index):
        # Retrieve data from the dataframe based on the index
        row  = self.patient_df.loc[index]
        genomics = torch.tensor(self.normalized_genomics.loc[index].values, dtype=torch.float32)
        dataset_name = row["dataset_name"]
        tissue_type_filter = self.datasets[dataset_name].tissue_type_filter
        slide_list = self.patient_dict[row[self.case_id_name]]
        patch_features, mask = self._load_wsi_embs_from_path(dataset_name, slide_list)
        label = row['label']
        if self.task_type == "Survival":
            censorship = row["censorship"]
            time = row["time"]
            label_names = ["time"]
        else:
            censorship = torch.tensor(0)
            time = torch.tensor(0)
            label_names = ["treatment_response"]


        data = {
                'input':{   
                            'patch_features': patch_features, 
                            'mask': mask,
                            'genomics':genomics
                        }, 
                'label': label, 
                'censorship': censorship, 
                'original_event_time': time,
                'label_names': label_names,
                'patient_id': row[self.case_id_name],
                'dataset_name': dataset_name,
            }
        return data

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.patient_df)

#####################################################################



data_frame = pd.read_csv(csv_file)
data_loader_config = {
  "datasets_configs": [TCGA_BRCA_dataset_config],
  "task_type": "Survival",
  "max_patches": 4096,
  "batch_size": 1,
  "real_batch_size": 8,
  "n_bins": 4,
  "sample": True,        # sample patches during train
  "test_sample": False,   # use all available patches during testing
  "load_slides_in_RAM": False,  # load in RAM patches for increasing data loading speed
  "label_name": "FUT",
  "censorships_name": "Survival",
  "eps": 0.000001,
  "num_workers": 1,
  "train_size": 0.7,
  "val_size": 0.15,
  "test_size": 0.15,
  "random_state": 42,
}
data_loader_config = Munch.fromDict(data_loader_config)
dataset = Multimodal_WSI_Genomic_Dataset(
    datasets_configs = data_loader_config.datasets_configs,
    task_type        = data_loader_config.task_type,
    max_patches      = data_loader_config.max_patches,
    n_bins           = data_loader_config.n_bins,
    eps              = data_loader_config.eps,
    sample           = data_loader_config.sample,
    load_slides_in_RAM = data_loader_config.load_slides_in_RAM
)


latest_val_cindex = []
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)



for fold in range(FOLDS):
    print(f'\n#########  Fold {fold + 1}/{FOLDS}  #########\n')

    # -----------------------------
    # Init data splits & loaders
    # -----------------------------
    print('\n[1] Init Loaders...', end=' ')
    train_patients, val_patients, test_patients = dataset.get_train_test_val_splits(
        train_size=data_loader_config.train_size,
        val_size=data_loader_config.val_size,
        test_size=data_loader_config.test_size,
        random_state=data_loader_config.random_state
    )

    train_loader, val_loader, test_loader = get_dataloaders(
        dataset=dataset,
        train_patients=train_patients,
        val_patients=val_patients,
        test_patients=test_patients,
        data_loader_config=data_loader_config
    )
    print('Done!')

    # -----------------------------
    # Training hyperparameters
    # -----------------------------
    LR = 1e-4
    WEIGHT_DECAY = 1e-4
    NUM_ACCUMULATION_STEPS = 8
    lambda_reg = 1e-4
    PATIENCE = 5

    # -----------------------------
    # Loss & regularization setup
    # -----------------------------
    print('\n[2] Init loss function...', end=' ')
    loss_fn = CrossEntropySurvLoss(alpha=0)

    if args.reg_type == 'omic':
        reg_fn = l1_reg_all
    elif args.reg_type == 'pathomic':
        reg_fn = l1_reg_modules
    else:
        reg_fn = None
    print('Done!')

    print(f'training: {len(train_loader)}, validation: {len(val_loader)}')

    # -----------------------------
    # Model setup
    # -----------------------------
    print('\n[3] Init Model...', end=' ')
    model = MCAT_Surv_SingleOmic(
        fusion='concat',
        n_classes=1
    ).to(device)
    print('Done!')

    # -----------------------------
    # Optimizer
    # -----------------------------
    print('\n[4] Init optimizer...', end=' ')
    optimizer = RAdam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    print('Done!')

    # -----------------------------
    # Early Stopping
    # -----------------------------
    print('\n[5] Setup EarlyStopping...', end=' ')
    early_stopping = EarlyStopping(
        warmup=0,
        patience=PATIENCE,
        stop_epoch=5,
        verbose=True
    )
    print('Done!')

    # -----------------------------
    # Monitor (C-index)
    # -----------------------------
    print('\n[6] Setup Validation C-Index Monitor...', end=' ')
    monitor_cindex = Monitor_CIndex()
    print('Done!\n')

    # -----------------------------
    # Training Loop
    # -----------------------------
    print('[7] Starting Training Loop...\n')
    for epoch in range(EPOCHS):
        single_train_loop_survival_coattn(
            epoch,
            model,
            train_loader,
            optimizer,
            2,
            loss_fn=loss_fn,
            reg_fn=reg_fn,
            lambda_reg=lambda_reg,
            gc=NUM_ACCUMULATION_STEPS,
            brca_label=args.label
        )

        stop = single_validate_survival_coattn(
            fold,
            epoch,
            model,
            val_loader,
            2,
            early_stopping=early_stopping,
            monitor_cindex=monitor_cindex,
            loss_fn=loss_fn,
            reg_fn=reg_fn,
            lambda_reg=lambda_reg,
            results_dir=results_dir,
            brca_label=args.label

        )

        if stop:
            print("→ Early stopping triggered!")
            break

    # -----------------------------
    # Save best model for fold
    # -----------------------------
    checkpoint_path = os.path.join(results_dir, f"MCAT_{fold}_checkpoint.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f'Model for Fold {fold + 1} saved to {checkpoint_path}')

print('\n Done!')

