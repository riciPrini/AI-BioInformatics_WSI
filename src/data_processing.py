import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_histogram(df):
    # Crea l'istogramma per 'fpkm_uq_unstranded'
    plt.figure(figsize=(8, 6))  # Imposta la dimensione della figura
    sns.histplot(df['fpkm_uq_unstranded'], kde=False, bins=21)  # Imposta bins e plot

    # Aggiungi etichette e titolo
    plt.title('fpkm_uq_unstranded')
    plt.xlabel('FPKM Unstranded')
    plt.ylabel('Conteggio')

    # Mostra il grafico
    plt.show()
# Percorso principale contenente le cartelle dei pazienti
base_path = './data2/'

# Lista per raccogliere i risultati
brca1 = []
brca2 = []

# Iterare attraverso tutte le cartelle e file
for root, dirs, files in os.walk(base_path):
    for file in files:
        if file.endswith('.tsv'):  # Verifica se il file è un .tsv
            file_path = os.path.join(root, file)
            print(file_path)
            try:
                # Leggi il file .tsv in un DataFrame pandas
                df = pd.read_csv(file_path, sep='\t', comment='#')
               
                if all(col in df.columns for col in ['gene_type', 'fpkm_uq_unstranded']):
                    # Filtra per BRCA1 e BRCA2
                    # filtered_df = df[df['gene_name'].isin(['BRCA1', 'BRCA2'])]
                    df_brca1 = df[df['gene_name'].isin(['BRCA1'])]
                    df_brca2 = df[df['gene_name'].isin(['BRCA2'])]

                    
                    for _, row in df_brca1.iterrows():
                        # print(row)
                        brca1.append({
                            'patient': os.path.basename(root),  # Nome della cartella del paziente
                            'gene_name': row['gene_name'],
                            'fpkm_uq_unstranded': row['fpkm_uq_unstranded']
                        })
                    for _, row in df_brca2.iterrows():
                        # print(row)
                        brca2.append({
                            'patient': os.path.basename(root),  # Nome della cartella del paziente
                            'gene_name': row['gene_name'],
                            'fpkm_uq_unstranded': row['fpkm_uq_unstranded']
                        })
            except Exception as e:
                print(f"Errore nella lettura del file {file_path}: {e}")
        

# Converti i risultati in un DataFrame
results_brca1 = pd.DataFrame(brca1)
results_brca2 = pd.DataFrame(brca2)
print(results_brca1)
print(results_brca2)
plot_histogram(results_brca1)
plot_histogram(results_brca2)


