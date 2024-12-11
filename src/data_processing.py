import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

base_path = './data/'

brca1_uq = []
brca2_uq = []

brca1 = []
brca2 = []

for root, dirs, files in os.walk(base_path):
    for file in files:
        
        if file.endswith('.tsv'):
            file_path = os.path.join(root, file)
            print(file_path)

            try:
                df = pd.read_csv(file_path, sep='\t', comment='#')
               
                if all(col in df.columns for col in ['gene_type', 'fpkm_uq_unstranded']):
                    # filtra per BRCA1 e BRCA2
                    df_brca1 = df[df['gene_name'].isin(['BRCA1'])]
                    df_brca2 = df[df['gene_name'].isin(['BRCA2'])]

                    for _, row in df_brca1.iterrows():
                        brca1.append({
                            'patient': os.path.basename(root), 
                            'gene_name': row['gene_name'],
                            'fpkm_unstranded': row['fpkm_unstranded']
                        })

                    for _, row in df_brca2.iterrows():
                        brca2.append({
                            'patient': os.path.basename(root),
                            'gene_name': row['gene_name'],
                            'fpkm_unstranded': row['fpkm_unstranded']
                        })

            except Exception as e:
                print(f"Errore nella lettura del file {file_path}: {e}")
        
def plot_histograms(df1, df2, df3, df4):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # 2 righe, 2 colonne
    
    # BRCA1 UQ
    sns.histplot(df1['fpkm_uq_unstranded'], kde=False, ax=axes[0, 0], color="blue")
    axes[0, 0].set_title('BRCA1 - FPKM UQ Unstranded')
    axes[0, 0].set_xlabel('Values')
    axes[0, 0].set_ylabel('Counts')

    # BRCA2 UQ
    sns.histplot(df2['fpkm_uq_unstranded'], kde=False, ax=axes[0, 1], color="green")
    axes[0, 1].set_title('BRCA2 - FPKM UQ Unstranded')
    axes[0, 1].set_xlabel('Values')
    axes[0, 1].set_ylabel('Counts')

    # BRCA1
    sns.histplot(df3['fpkm_unstranded'], kde=False, ax=axes[1, 0], color="blue")
    axes[1, 0].set_title('BRCA1 - FPKM Unstranded')
    axes[1, 0].set_xlabel('Values')
    axes[1, 0].set_ylabel('Counts')

    # BRCA2
    sns.histplot(df4['fpkm_unstranded'], kde=False, ax=axes[1, 1], color="green")
    axes[1, 1].set_title('BRCA2 - FPKM Unstranded')
    axes[1, 1].set_xlabel('Values')
    axes[1, 1].set_ylabel('Counts')

    # Aggiusta il layout per evitare sovrapposizioni
    plt.tight_layout()
    plt.show()


results_brca1_uq = pd.DataFrame(brca1_uq)
results_brca2_uq = pd.DataFrame(brca2_uq)

results_brca1 = pd.DataFrame(brca1)
results_brca2 = pd.DataFrame(brca2)

plot_histograms(results_brca1_uq, results_brca2_uq, results_brca1, results_brca2)
