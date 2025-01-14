# from pytorch_model_summary import summary
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
# from lifelines.statistics import logrank_test
from MIL import *

## Dataset Creation
def get_dataloaders(dataset, train_patients, val_patients, test_patients, data_loader_config):
    mask = np.isin(train_patients, dataset.patient_df.index)
    # Filter the array to keep only elements in df.index
    filtered_train_patients = train_patients[mask]
    if len(filtered_train_patients) != len(train_patients):
        print("Some train patients are not in the dataset: ", set(train_patients) - set(filtered_train_patients))
    prefetch_factor = 4
    if data_loader_config.num_workers == 0:
        prefetch_factor = None
    train_dataloader = DataLoader(
                                Subset(dataset, filtered_train_patients), 
                                batch_size=data_loader_config.batch_size, 
                                shuffle=True, 
                                drop_last=True, 
                                pin_memory=True, 
                                num_workers=data_loader_config.num_workers, 
                                prefetch_factor=prefetch_factor
                            )
    if val_patients is not None:
        mask = np.isin(val_patients, dataset.patient_df.index)
        filtered_val_patients = val_patients[mask]
        if len(filtered_val_patients) != len(val_patients):
            print("Some val patients are not in the dataset: ", set(val_patients) - set(filtered_val_patients))
        batch_size = data_loader_config.batch_size
        if data_loader_config.test_sample == False:
            batch_size = 1
        val_dataloader = DataLoader(
                                        Subset(dataset, filtered_val_patients), 
                                        batch_size=batch_size,
                                        shuffle=False, 
                                        drop_last=False, 
                                        pin_memory=True, 
                                        num_workers=data_loader_config.num_workers, 
                                        prefetch_factor=prefetch_factor,
                                )
    else:
        val_dataloader = None   
    if test_patients is not None:
        mask = np.isin(test_patients, dataset.patient_df.index)
        filtered_test_patients = test_patients[mask]
        if len(filtered_test_patients) != len(test_patients):
            print("Some test patients are not in the dataset: ", set(test_patients) - set(filtered_test_patients))
        batch_size = data_loader_config.batch_size
        if data_loader_config.test_sample == False:
            batch_size = 1
        test_dataloader = DataLoader(
                                        Subset(dataset, filtered_test_patients), 
                                        batch_size=batch_size, 
                                        shuffle=False, 
                                        drop_last=False, 
                                        pin_memory=True, 
                                        num_workers=data_loader_config.num_workers, 
                                        prefetch_factor=prefetch_factor,
                                    )
    else:
        test_dataloader = None
    return train_dataloader, val_dataloader, test_dataloader

