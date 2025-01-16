
import os
import torch
from torch import nn
import torch.optim as optim
from torch.backends import cudnn
import torch.nn.init as init
# from pytorch_model_summary import summary
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
import time
import logging
from tqdm import tqdm
import math
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from munch import Munch
# from sksurv.metrics import concordance_index_censored
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
# from lifelines import KaplanMeierFitter
# from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
# from lifelines.statistics import logrank_t
current_dir = os.path.dirname(os.path.abspath(__file__))
TCGA_BRCA_dataset_config = {
    "name": "TCGA_BRCA",
    "parameters": {
      "dataframe_path": os.path.join(current_dir, "../dataset/dataset_brca.csv"),
      "pt_files_path": os.path.join(current_dir,"../LAB_WSI_Genomics/TCGA_BRCA/Data/wsi/features_UNI/pt_files"),
      "genomics_path": os.path.join(current_dir, "../dataset/tpm_unstranded.csv"), # fpkm_uq_unstranded # fpkm_unstranded # tpm_unstranded # unstranded
      "tissue_type_filter": [],
      "label_name": "FUT",
      "censorships_name": "Survival",
      "case_id_name": "case_id",
      "slide_id_name": "slide_id",
     
    }
}
TCGA_BRCA_dataset_config = Munch.fromDict(TCGA_BRCA_dataset_config)


class Multimodal_WSI_Genomic_Dataset(Dataset):
    def __init__(self,  datasets_configs = [TCGA_BRCA_dataset_config],
                        task_type="Survival", # Survival or treatment_response
                        max_patches=4096,
                        n_bins=4,
                        eps=1e-6,
                        sample=True,
                        load_slides_in_RAM=False,
                        ):
        self.task_type = task_type
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
        # print(row)
        # brca1_value = genomics[self.normalized_genomics.columns.get_loc("BRCA1")]
        # brca2_value = genomics[self.normalized_genomics.columns.get_loc("BRCA2")]
        tissue_type_filter = self.datasets[dataset_name].tissue_type_filter
        slide_list = self.patient_dict[row[self.case_id_name]]
        patch_features, mask = self._load_wsi_embs_from_path(dataset_name, slide_list)
        label = row['label']
        # print(label)
        brca1 = row['BRCA1']
        brca2 = row['BRCA2']
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
                            'genomics':genomics, # questa roba va nella forwardM
                            'brca_1': brca1,
                            'brca_2': brca2
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


class ABMIL_Multimodal(nn.Module):
    def __init__(self,
                     input_dim=1024,
                     genomics_input_dim = 19962,
                     inner_dim=64, 
                     output_dim=4, 
                     use_layernorm=False, 
                     input_modalities = ["WSI", "Genomics"],
                     genomics_dropout = 0.5,
                     dropout=0.0,
                ):
        super(ABMIL_Multimodal,self).__init__()

        self.inner_proj = nn.Linear(input_dim,inner_dim)
        self.output_dim = output_dim
        self.device="cuda" if torch.cuda.is_available() else "cpu"
        self.input_modalities = input_modalities
        self.use_layernorm = use_layernorm
        self.dropout = nn.Dropout(dropout)
        if self.use_layernorm:
            self.layernorm = nn.LayerNorm(inner_dim)
        self.attention_V = nn.Linear(inner_dim, inner_dim)
        self.attention_U = nn.Linear(inner_dim, inner_dim)
        self.sigmoid = nn.Sigmoid()
        self.attention_weights = nn.Linear(inner_dim, 1)

        self.genomics_dropout = nn.Dropout(genomics_dropout)
        self.fc_genomics = nn.Sequential(
                                            nn.Linear(genomics_input_dim, inner_dim),
                                            nn.ReLU(),
                                            nn.Linear(inner_dim, inner_dim),
                                        )                                                        

        # Output layer
        final_layer_input_dim = 0
        if "WSI" in input_modalities:
            final_layer_input_dim += inner_dim
        if "Genomics" in input_modalities:
            final_layer_input_dim += inner_dim
            
        self.output_layer = nn.Linear(final_layer_input_dim, output_dim)
        self.braca1_output_layer = nn.Linear(final_layer_input_dim, 1)  # BRCA1 (binary classification)
        self.braca2_output_layer = nn.Linear(final_layer_input_dim, 1)  # BRCA2 (binary classification)
        
    def forward(self, data):
        # Extract patch features
        if "WSI" in self.input_modalities:
            x = data['patch_features']  # x is a dictionary with key 'patch_features'
            mask = data['mask']
            x = x[~mask.bool()].unsqueeze(0)
            x = self.inner_proj(x)
            
            if self.use_layernorm:
                x = self.layernorm(x)        
            
            # Apply attention mechanism
            V = torch.tanh(self.attention_V(x))  # Shape: (batch_size, num_patches, inner_dim)
            U = self.sigmoid(self.attention_U(x))  # Shape: (batch_size, num_patches, inner_dim)
            
            # Compute attention scores
            attn_scores = self.attention_weights(V * U)  # Shape: (batch_size, num_patches, 1)
            attn_scores = torch.softmax(attn_scores, dim=1)  # Shape: (batch_size, num_patches, 1)
            
            # Weighted sum of patch features
            weighted_sum = torch.sum(attn_scores * x, dim=1)  # Shape: (batch_size, inner_dim)
            weighted_sum = self.dropout(weighted_sum)

            # Final WSI embedding
            wsi_embedding = weighted_sum

        if "Genomics" in self.input_modalities:
            genomics = data["genomics"]
            genomics = self.genomics_dropout(genomics)
            # Final Genomic embedding
            genomics_embedding = self.fc_genomics(genomics)

        if "WSI" in self.input_modalities and "Genomics" in self.input_modalities:
            x = torch.cat([wsi_embedding,genomics_embedding], dim=1)
        elif "WSI" in self.input_modalities:
            x = wsi_embedding
        elif "Genomics" in self.input_modalities:
            x = genomics_embedding
        
        output = self.output_layer(x)  # Shape: (batch_size, output_dim)
        braca1_prediction = self.braca1_output_layer(x)
        braca2_prediction = self.braca2_output_layer(x)
        return output, braca1_prediction, braca2_prediction

class NLLSurvLoss(nn.Module):
    """
    The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020).
    Code borrowed from https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py
    Parameters
    ----------
    alpha: float
        TODO: document
    eps: float
        Numerical constant; lower bound to avoid taking logs of tiny numbers.
    reduction: str
        Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum']
    """
    def __init__(self, alpha=0.0, eps=1e-7, reduction='sum'):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.reduction = reduction

    def __call__(self, h, y, t, c):
        """
        Parameters
        ----------
        h: (n_batches, n_classes)
            The neural network output discrete survival predictions such that hazards = sigmoid(h).
        y_c: (n_batches, 2) or (n_batches, 3)
            The true time bin label (first column) and censorship indicator (second column).
        """

        return nll_loss(h=h, y=y, c=c,
                        alpha=self.alpha, eps=self.eps,
                        reduction=self.reduction)


# TODO: document better and clean up
def nll_loss(h, y, c, alpha=0.0, eps=1e-7, reduction='sum'):
        """
        The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020).
        Code borrowed from https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py
        Parameters
        ----------
        h: (n_batches, n_classes)
            The neural network output discrete survival predictions such that hazards = sigmoid(h).
        y: (n_batches, 1)
            The true time bin index label.
        c: (n_batches, 1)
            The censoring status indicator.
        alpha: float
            The weight on uncensored loss 
        eps: float
            Numerical constant; lower bound to avoid taking logs of tiny numbers.
        reduction: str
            Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum']
        References
        ----------
        Zadeh, S.G. and Schmid, M., 2020. Bias in cross-entropy-based training of deep survival networks. IEEE transactions on pattern analysis and machine intelligence.
        """
        # print("h shape", h.shape)

        # make sure these are ints
        y = y.type(torch.int64)
        c = c.type(torch.int64)

        hazards = torch.sigmoid(h)
        # print("hazards shape", hazards.shape)

        S = torch.cumprod(1 - hazards, dim=1)
        # print("S.shape", S.shape, S)

        S_padded = torch.cat([torch.ones_like(c), S], 1)
        # S(-1) = 0, all patients are alive from (-inf, 0) by definition
        # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
        # hazards[y] = hazards(1)
        # S[1] = S(1)
        # TODO: document and check

        # print("S_padded.shape", S_padded.shape, S_padded)


        # TODO: document/better naming
        s_prev = torch.gather(S_padded, dim=1, index=y).clamp(min=eps)
        h_this = torch.gather(hazards, dim=1, index=y).clamp(min=eps)
        s_this = torch.gather(S_padded, dim=1, index=y+1).clamp(min=eps)
        # print('s_prev.s_prev', s_prev.shape, s_prev)
        # print('h_this.shape', h_this.shape, h_this)
        # print('s_this.shape', s_this.shape, s_this)

        # c = 1 means censored. Weight 0 in this case 
        uncensored_loss = -(1 - c) * (torch.log(s_prev) + torch.log(h_this))
        censored_loss = - c * torch.log(s_this)


        # print('uncensored_loss.shape', uncensored_loss.shape)
        # print('censored_loss.shape', censored_loss.shape)

        neg_l = censored_loss + uncensored_loss
        if alpha is not None:
            loss = (1 - alpha) * neg_l + alpha * uncensored_loss

        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
        else:
            raise ValueError("Bad input for reduction: {}".format(reduction))

        return loss