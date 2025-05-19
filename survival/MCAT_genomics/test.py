#10e_brca1 [0.6639650872817955, 0.5991271820448878, 0.594139650872818, 0.6041147132169576, 0.6271820448877805]
#10e_brca2 [0.6365336658354115, 0.6165835411471322, 0.6290523690773068, 0.6446384039900249, 0.6783042394014963]
#15e_brca1 [0.6159600997506235, 0.6315461346633416, 0.6296758104738155, 0.6066084788029925, 0.6128428927680798]
#15e_brca2 [0.662718204488778, 0.6664588528678305, 0.5953865336658354, 0.6165835411471322, 0.6683291770573566]
# --- BRCA1 --- #
# tpm_unstranded [0.6639650872817955, 0.5991271820448878, 0.594139650872818, 0.6041147132169576, 0.6271820448877805]
# unstranded [0.6620947630922693, 0.6184538653366584, 0.6066084788029925, 0.6346633416458853, 0.6334164588528678]
# fpkm_unstranded [0.669576059850374, 0.6072319201995012, 0.5972568578553616, 0.6253117206982544, 0.6203241895261845]
# fpkm_uq_unstranded [0.6770573566084788, 0.5879052369077307, 0.5972568578553616, 0.6465087281795511, 0.6340399002493765]
# --- BRCA2 --- #
# tpm_unstranded [0.662718204488778, 0.6109725685785536, 0.60785536159601, 0.6602244389027432, 0.64214463840399]
# unstranded [0.6571072319201995, 0.6084788029925187, 0.6053615960099751, 0.6433915211970075, 0.628428927680798]
# fpkm_unstranded [0.6683291770573566, 0.6122194513715711, 0.600997506234414, 0.6577306733167082, 0.6446384039900249]
# fpkm_uq_unstranded [0.6677057356608479, 0.6184538653366584, 0.6059850374064838, 0.6652119700748129, 0.6271820448877805]
import argparse
from torch.utils.data import Dataset 
import warnings
from munch import Munch
import os
import random
import torch
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from utils.core_utils import summary_survival,test_survival
from models.model_coattn import MCAT_Surv_SingleOmic
from models.dataset_survival import *
warnings.filterwarnings("ignore") 

parser = argparse.ArgumentParser(description='BRCA overexprection')

parser.add_argument('--seed', '-s', type=int, default=7,
                        help='Random seed')
parser.add_argument('--data_directory', type=str, default='/work/ai4bio2024/brca_surv/LAB_WSI_Genomics/TCGA_BRCA/Data/wsi/features_UNI/pt_files',
                        help='Dataset directory')
parser.add_argument('--labels_file', type=str, default='/work/ai4bio2024/brca_surv/dataset/genomics_csv/dataset_survival_unstranded.csv',
                        help='label file path')
parser.add_argument('--label', type=str, default='BRCA2',
                        help='Label to use for training')
parser.add_argument('--folds', '-f', type=int, default=5,
                    help='Number of folds')
parser.add_argument('--epoch', '-e', type=int, default=10,
                    help='Number of epochs')                  

args = parser.parse_args()

SEED = args.seed
FOLDS = args.folds
MODEL = "MCAT"

results_dir = f"/work/ai4bio2024/brca_surv/survival/MCAT_genomics/checkpoint/unstranded/"
csv_file = args.labels_file
data_dir = args.data_directory

TCGA_BRCA_dataset_config = {
    "name": "TCGA_BRCA",
    "parameters": {
      "dataframe_path": csv_file,
      "pt_files_path": data_dir,
      "genomics_path": "/work/ai4bio2024/brca_surv/LAB_WSI_Genomics/TCGA_BRCA/Data/genomic/unstranded.csv", # fpkm_uq_unstranded # fpkm_unstranded # tpm_unstranded # unstranded
      "tissue_type_filter": [],
      "label_name": "FUT",
      "censorships_name": "Survival",
      "case_id_name": "case_id",
      "slide_id_name": "slide_id",
    }
}
TCGA_BRCA_dataset_config = Munch.fromDict(TCGA_BRCA_dataset_config)
def setup(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # If using CUDA.
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


#####################################################################

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
                'slide_id':row['slide_id']
            }
        return data

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.patient_df)

#####################################################################


setup(SEED)

if torch.cuda.is_available():
    device = 'cuda'
    print(f'# CUDA available: {torch.cuda.get_device_name(0)}')
else:
    device = 'cpu'
    print('No GPU!')

print(f'Testing overall survival - {args.epoch}e_{args.label}\n')

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


c_indexes = []
for f in range(FOLDS):

    print(f'######## Testing model {f+1} ########\n')

    print('Initializing data loaders...', end=' ')
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

    print('\nInit Model...', end=' ')
    model_dict = {'fusion': 'concat','n_classes': 1} 
    model = MCAT_Surv_SingleOmic(**model_dict).to(device)
    print('Done!')

    # --- Load checkpoint ---
    model.load_state_dict(torch.load(os.path.join(results_dir, "MCAT_{}_checkpoint.pth".format(f))),strict=False)
    
    # --- Evaluate on Validation Set ---
    results_val_dict, val_cindex = test_survival(model, val_loader, 2, args.label)
    print('Val c-Index: {:.4f}'.format(val_cindex))
    
    c_indexes.append(val_cindex)

    # for k,v in results_val_dict.items():
    #     print(k,v)
print(c_indexes)
