import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import gaussian_kde
import os

import warnings
warnings.filterwarnings('ignore')

TPM = pd.read_csv('/work/ai4bio2024/brca_surv/LAB_WSI_Genomics/TCGA_BRCA/Data/genomic/tpm_unstranded.csv', sep='\t')

TPM = TPM.iloc[:, [0]].join(TPM[['ENSG00000012048.23', 'ENSG00000139618.16']])

TPM = TPM.rename(columns = {'Unnamed: 0':'case_id', 'ENSG00000012048.23': 'BRCA1', 'ENSG00000139618.16':'BRCA2'})

# Merge dataset

dataset = pd.read_csv('/work/ai4bio2024/brca_surv/LAB_WSI_Genomics/TCGA_BRCA/TCGA_BRCA_labels_multimodal.tsv', sep='\t')

dataset = pd.merge(dataset, TPM, on='case_id', how='inner')

dataset.to_csv(f'./dataset_brca_regression.csv', index=False)

wsi = pd.read_csv('./wsi_id.csv')

result = pd.merge(dataset, wsi, left_on='slide_id', right_on='slide_id', how='left')

result.to_csv('./dataset_brca_wsi_regression.csv',index=False)