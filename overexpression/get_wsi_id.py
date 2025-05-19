import os
import pandas as pd

path ='/work/h2020deciderficarra_shared/TCGA/BRCA/data'

data = []

for foldername in os.listdir(path):
    folder_path = os.path.join(path, foldername)
    
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith('.svs'):
                data.append({'id': foldername, 'slide_id': os.path.splitext(filename)[0]})

df = pd.DataFrame(data)
df.to_csv('./wsi_id.csv', index=False)