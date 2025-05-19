import pandas as pd
import argparse
import os

## TCGA-OL-A5RW-01Z-00-DX1.E16DE8EE-31AF-4EAF-A85F-DB3E3E2C3BFF
def file_exists(slide_id):
    slide_id_no_ext = slide_id.rstrip('.svs')  # Rimuove ".svs" dalla stringa
    # print(slide_id_no_ext)
    return os.path.exists(f'/work/h2020deciderficarra_shared/TCGA/BRCA/features_UNI/pt_files/{slide_id_no_ext}.pt')

slide_data = pd.read_csv('/homes/rprini/MCAT_custom/dataset_csv_sig/filtered/my_tcga_brca_all_clean.csv.zip', compression='zip', header=0, index_col=0, sep=',',  low_memory=False)
# slide_data = pd.read_csv('/homes/rprini/MCAT_custom/dataset_csv_sig/filtered/my_tcga_brca_all_clean.csv.zip', compression='zip', header=0, index_col=0, sep=',',  low_memory=False)
output_path =  "/homes/rprini/MCAT_custom/dataset_csv_sig/filtered/only_brca.csv.zip"
# slide_id = slide_data["slide_id"].values[0].rstrip('.svs')
# df = slide_data[slide_data['slide_id'].apply(file_exists)]
print(slide_data)
# # Seleziona prime 9 colonne per posizione (.iloc), e specifiche colonne per nome

''' -- Create brca_only --  '''
# df_filtered = slide_data.iloc[:, :9].copy()
# selected_extra = slide_data[['BRCA1_rnaseq', 'BRCA2_rnaseq']]
# print(selected_extra.head())
# df = pd.concat([df_filtered, selected_extra], axis=1)
# print(df)
# print(df.columns)
# df.to_csv(output_path, index=True, compression={'method': 'zip'})


# parser = argparse.ArgumentParser(description='BRCA overexprection')
# parser.add_argument('--id', type=str, default='TCGA-AN-A0XW', help='id')
# args = parser.parse_args()
# for i in range(5):
#     fname = f'splits_{i}.csv'
#     df = pd.read_csv(f'/homes/rprini/MCAT/splits_ref/{fname}')
#     df = df[df['train'] != args.id]
#     df.reset_index(drop=True, inplace=True)
#     df.to_csv(f'/homes/rprini/MCAT/splits_ref/{fname}',index=False)
# print(df)