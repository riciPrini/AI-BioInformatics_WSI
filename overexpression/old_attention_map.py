import torch
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from models import *
from PIL import Image
import numpy as np
import h5py
import cv2
import openslide
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='BRCA Attention Map')
parser.add_argument('--mid', '-m', type=int, default=1,
                        help='ID of the model used for attention over BRCA overexprection')
parser.add_argument('--sid', '-s', type=str, default='TCGA-E2-A155',
                        help='Identifier of the patient to be used for computing the attention map')

args = parser.parse_args()



def get_coordinates(original_coord, original_dim, new_dim):
    x_orig, y_orig = original_coord
    w_orig, h_orig = original_dim
    w_new, h_new = new_dim
    

    downsampling_width = w_orig / w_new
    downsampling_height = h_orig / h_new
    
    x_new = int(x_orig / downsampling_width)
    y_new = int(y_orig / downsampling_height)
    
    return (x_new, y_new)

if torch.cuda.is_available():
    device = 'cuda'
    print(f'# CUDA available: {torch.cuda.get_device_name(0)}')
else:
    device = 'cpu'
    print('No GPU!')


#model = ABMIL()
model = DS_ABMIL()
model.load_state_dict(torch.load(f'./model_weights_{args.mid}.pth', weights_only=True))
model = model.to(device)
model.eval()

patient_id = args.sid#'TCGA-E2-A155'
dataframe = pd.read_csv("/work/ai4bio2024/brca_surv/dataset/dataset_brca_wsi.csv")
patient_row = dataframe.loc[dataframe["case_id"] == patient_id].iloc[0]
id_ = patient_row["id"]
slide_id = patient_row["slide_id"]
gene = 'BRCA1'
ground_truth = patient_row[gene]
print(f'Ground Trunth label of {patient_id} is {ground_truth}')

wsi_path = f'/work/h2020deciderficarra_shared/TCGA/BRCA/data/{id_}/{slide_id}.svs'
slide = openslide.OpenSlide(wsi_path)
# Ottieni le dimensioni dell'immagine
width, height= slide.dimensions

mask_img_path  = f'/work/h2020deciderficarra_shared/TCGA/BRCA/patches_CLAM/masks/{slide_id}.jpg'
mask_img = np.array(Image.open(mask_img_path))
height_mask, width_mask  = mask_img.shape[:2]


original_dim = (width, height)
new_dim = (width_mask, height_mask)


#h5_file_path = f"/work/h2020deciderficarra_shared/TCGA/BRCA/patches_CLAM/patches/{slide_id}.h5"
h5_file_path = f"/work/h2020deciderficarra_shared/TCGA/BRCA/features_UNI/h5_files/{slide_id}.h5"
with h5py.File(h5_file_path, 'r') as f:
    coordinates = f['coords'][:]

uni_file_path = f'/work/h2020deciderficarra_shared/TCGA/BRCA/features_UNI/pt_files/{slide_id}.pt'
data = torch.load(uni_file_path, weights_only=True)
data = data.unsqueeze(0)
with torch.no_grad():
    data = data.to(device)
    output , attn_scores = model(data)

if isinstance(output, tuple):
    output = output[0]

if output > 0.5:
    output = 1
else:
    output = 0
print(f'The predicted label of {patient_id} is {output}')

attn_scores = attn_scores.squeeze().cpu().detach().numpy()
attn_scores = (attn_scores - attn_scores.min()) / (attn_scores.max() - attn_scores.min()) ## normalize



patch_dim = round(1024*width_mask/width)

heatmap = np.zeros((height_mask, width_mask, 3), dtype=np.uint8)
alpha = 0.4
for i, coord in enumerate(coordinates):
    x, y = get_coordinates(coord,original_dim,new_dim)
    x_end = min(x + patch_dim, width_mask)
    y_end = min(y + patch_dim, height_mask)
    color = cm.jet(attn_scores[i])[:3]
    color = (np.array(color)*255).astype(np.uint8)
    heatmap[y:y_end,x:x_end] = color
    new_patch = (1 - alpha) * mask_img[y:y_end,x:x_end] + alpha * heatmap[y:y_end,x:x_end]
    mask_img[y:y_end,x:x_end] = np.clip(new_patch, 0, 255).astype(np.uint8)

img_with_heatmap = Image.fromarray(mask_img)

img_with_heatmap.save("old_attention_map_val.png")

