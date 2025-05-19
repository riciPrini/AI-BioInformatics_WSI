import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import h5py
import openslide
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter

from models.model_coattn import MCAT_Surv
from datasets.dataset_survival import Generic_MIL_Survival_Dataset

import warnings
warnings.filterwarnings("ignore")

# Funzioni utili
def get_coordinates(original_coord, original_dim, new_dim):
    x_orig, y_orig = original_coord
    w_orig, h_orig = original_dim
    w_new, h_new = new_dim
    
    downsampling_width = w_orig / w_new
    downsampling_height = h_orig / h_new
    
    x_new = int(x_orig / downsampling_width)
    y_new = int(y_orig / downsampling_height)
    
    return (x_new, y_new)

def get_attention_map(model, image_tensor, layer='layer4'):
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    if hasattr(model, layer):
        hook_handle = getattr(model, layer).register_forward_hook(get_activation(layer))

        with torch.no_grad():
            _ = model(image_tensor)

        hook_handle.remove()

        return activation[layer]
    else:
        raise ValueError(f'{layer} doesn\'t exists.')

def get_patch(wsi, coord, patch_dim=1024):
    x, y = coord
    level = 0
    patch = wsi.read_region((x, y), level, (patch_dim, patch_dim))

    patch_np = np.array(patch)
    patch_tensor = torch.from_numpy(patch_np).permute(2, 0, 1)
    if patch_tensor.size(0) == 4:
        patch_tensor = patch_tensor[:3, :, :]
    patch_tensor = patch_tensor.float() / 255.0
    return patch_tensor

def get_heatmap(model, wsi, coord, out_dim, att, pre_processor):
    patch = get_patch(wsi, coord)

    if isinstance(patch, torch.Tensor):
        to_pil = transforms.ToPILImage()
        img = to_pil(patch)

    # Process on CPU
    tensor = pre_processor(img).unsqueeze(0).to(cpu_device)  # CPU instead of GPU
    attention_map = get_attention_map(model, tensor)
    attention_map = attention_map.squeeze().cpu().numpy()

    if isinstance(attention_map, np.ndarray):
        attention_map = torch.tensor(attention_map)
    
    attention_map = torch.mean(attention_map, axis=0)
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())

    attention_map *= att
    attention_map = attention_map.unsqueeze(0).unsqueeze(0)
    attention_map = F.interpolate(attention_map, size=out_dim, mode='bilinear', align_corners=False)
    attention_map = attention_map.squeeze().cpu().numpy()

    return attention_map

# Config iniziale
device = 'cuda'

# Streamlit UI
st.title("BRCA - Survival Prediction & Attention Map Demo")

uploaded_model = st.file_uploader("Upload your model checkpoint (.pt)", type=['pt'])

patient_id = st.text_input("Enter Patient ID", "TCGA-E2-A155")

if st.button("Run Prediction"):
    if uploaded_model is not None and patient_id:
        st.write("Loading... please wait!")

        # Carica dataset
        dataset = Generic_MIL_Survival_Dataset(
            csv_path="/homes/rprini/MCAT_custom/dataset_csv_sig/filtered/my_tcga_brca_all_clean.csv.zip",
            mode='coattn', 
            apply_sig=True,
            data_dir='/work/h2020deciderficarra_shared/TCGA/BRCA/features_UNI/',
            shuffle=False, 
            seed=1, 
            print_info=False,
            patient_strat=False, 
            n_bins=4,
            label_col='survival_months', ignore=[]
        )
        train_dataset, val_dataset = dataset.return_splits(from_id=False,
            csv_path='/homes/rprini/MCAT/splits/5foldcv/tcga_brca/splits_0.csv')

        model_dict = {"omic_sizes": [91, 353, 553, 480, 1565, 480]}
        model = MCAT_Surv(**model_dict)
        # model.load_state_dict(torch.load(uploaded_model, weights_only=True))
        model.load_state_dict(torch.load(uploaded_model, map_location=device), strict=False)
        # model.load_state_dict(torch.load(uploaded_model, map_location=device), strict=False)
        model = model.to(device)
        model.eval()

        # Carica paziente
        dataframe = pd.read_csv("/work/ai4bio2024/brca_surv/dataset/dataset_brca_wsi.csv")
        patient_row = dataframe[dataframe["case_id"] == patient_id].iloc[0]
        id_ = patient_row["id"]
        slide_id = patient_row["slide_id"]

        wsi_path = f'/work/h2020deciderficarra_shared/TCGA/BRCA/data/{id_}/{slide_id}.svs'
        slide = openslide.OpenSlide(wsi_path)
        width, height = slide.dimensions

        mask_img_path = f'/work/h2020deciderficarra_shared/TCGA/BRCA/patches_CLAM/masks/{slide_id}.jpg'
        mask_img = np.array(Image.open(mask_img_path))
        height_mask, width_mask = mask_img.shape[:2]

        original_dim = (width, height)
        new_dim = (width_mask, height_mask)

        h5_file_path = f"/work/h2020deciderficarra_shared/TCGA/BRCA/features_UNI/h5_files/{slide_id}.h5"
        with h5py.File(h5_file_path, 'r') as f:
            coordinates = f['coords'][:]

        data = torch.load(f'/work/h2020deciderficarra_shared/TCGA/BRCA/features_UNI/pt_files/{slide_id}.pt', map_location=device)
        data = data.unsqueeze(0)

        
        
        cpu_device = 'cuda'
        resnet = torchvision.models.resnet50(pretrained=True)
        resnet.eval()
        resnet = resnet.to(cpu_device)

        idx = 141  # esempio indice del paziente
        (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c) = val_dataset.__getitem__(idx)
        
        with torch.no_grad():
            data = data.to(device)
            data_omic1 = data_omic1.type(torch.FloatTensor).to(device)
            data_omic2 = data_omic2.type(torch.FloatTensor).to(device)
            data_omic3 = data_omic3.type(torch.FloatTensor).to(device)
            data_omic4 = data_omic4.type(torch.FloatTensor).to(device)
            data_omic5 = data_omic5.type(torch.FloatTensor).to(device)
            data_omic6 = data_omic6.type(torch.FloatTensor).to(device)
            hazards, survival, output, attn = model(x_path=data,x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)


        survival = survival.cpu().numpy()
        bins = np.array([3.8, 24.0, 46.0, 77.9, 244.9])
        mid_bin_times = (bins[:-1] + bins[1:]) / 2
        expected_survival = (survival * mid_bin_times).sum()

        st.success(f"Predicted Overall Survival: {expected_survival:.2f} months")

        attn_scores = attn["coattn"]
        attn_scores = attn_scores.squeeze().cpu().detach().numpy()
        attn_scores = np.mean(attn_scores, axis=0)
        attn_scores = (attn_scores - attn_scores.min()) / (attn_scores.max() - attn_scores.min())

        pre_preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        patch_dim = round(1024 * width_mask / width)

        heatmap = np.zeros((height_mask, width_mask))
        alpha = 0.6

        for i, coord in enumerate(coordinates):
            x, y = get_coordinates(coord, original_dim, new_dim)
            x_end = min(x + patch_dim, width_mask)
            y_end = min(y + patch_dim, height_mask)

            out_dim = (y_end - y, x_end - x)
            att = get_heatmap(resnet, slide, coord, out_dim, attn_scores[i], pre_preprocess)
            heatmap[y:y_end, x:x_end] = att

        heatmap = gaussian_filter(heatmap, sigma=2)
        mask = heatmap != 0
        if mask.any():  # Check if there are any non-zero values
            min_att = np.min(heatmap[mask])
            heatmap[mask] = (heatmap[mask] - min_att) / (heatmap.max() - min_att)
        
        threshold = 0.2
        mask_heatmap = heatmap < threshold
        heatmap[mask_heatmap] = 0

        colors = cm.jet(heatmap)[:, :, :3]
        colors = (colors * 255).astype(np.uint8)

        for i, coord in enumerate(coordinates):
            x, y = get_coordinates(coord, original_dim, new_dim)
            x_end = min(x + patch_dim, width_mask)
            y_end = min(y + patch_dim, height_mask)
        
            # Apply heatmap only where values are greater than 0
            new_patch = np.where(np.expand_dims(heatmap[y:y_end, x:x_end] > 0, axis=-1),
                            (1 - alpha) * mask_img[y:y_end, x:x_end] + alpha * colors[y:y_end, x:x_end],
                            mask_img[y:y_end, x:x_end])
            mask_img[y:y_end, x:x_end] = np.clip(new_patch, 0, 255).astype(np.uint8)

        st.image(mask_img, caption="Generated Attention Map", use_column_width=True)
