import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from models.model_coattn import MCAT_Surv
from PIL import Image
import numpy as np
import h5py
import openslide
import pandas as pd
from scipy.ndimage import gaussian_filter
import argparse
import warnings
from datasets.dataset_survival import Generic_WSI_Survival_Dataset, Generic_MIL_Survival_Dataset

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description='BRCA Attention Map')
parser.add_argument('--mid', '-m', type=int, default=0,
                        help='ID of the model usef for attention over BRCA overexprection')
parser.add_argument('--mode', type=str, choices=['omic', 'path', 'pathomic', 'cluster', 'coattn'], default='coattn', help='Specifies which modalities to use / collate function in dataloader.')
parser.add_argument('--apply_sig', action='store_true', default=True, help='Use genomic features as signature embeddings.')
parser.add_argument('--data_root_dir',   type=str, default='/work/h2020deciderficarra_shared/TCGA/BRCA/features_UNI/', help='Data directory to WSI features (extracted via CLAM')
parser.add_argument('--results_dir',     type=str, default='/homes/rprini/results', help='Results directory (Default: ./results)')
parser.add_argument('--seed', 			 type=int, default=1, help='Random seed for reproducible experiment (default: 1)')
parser.add_argument('--split_dir',       type=str, default='/homes/rprini/MCAT/splits/5foldcv/tcga_brca', help='Which cancer type within ./splits/<which_splits> to use for training. Used synonymously for "task" (Default: tcga_blca_100)')

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

# Main model on GPU
device = 'cuda'
dataset = Generic_MIL_Survival_Dataset(csv_path = "/homes/rprini/MCAT_custom/dataset_csv_sig/filtered/my_tcga_brca_all_clean.csv.zip",
                                       mode = args.mode,
                                       apply_sig = args.apply_sig,
                                       data_dir= args.data_root_dir,
                                       shuffle = False, 
                                       seed = args.seed, 
                                       print_info = False,
                                       patient_strat= False,
                                       n_bins=4,
                                       label_col = 'survival_months',
                                       ignore=[])
train_dataset, val_dataset = dataset.return_splits(from_id=False, 
                csv_path='{}/splits_{}.csv'.format(args.split_dir, 0))

model_dict = {"omic_sizes": [91, 353, 553, 480, 1565, 480]}

model = MCAT_Surv(**model_dict)
model.load_state_dict(torch.load(f'/homes/rprini/MCAT/results/5foldcv/all/MCAT_nll_surv_a0.0_5foldcv_gc32_concat/tcga_brca/_MCAT_nll_surv_a0.0_5foldcv_gc32_concat_s1/s_2_checkpoint.pt', map_location=device), strict=False)

# model.load_state_dict(torch.load(f'/homes/rprini/MCAT/results/5foldcv/distillation/MCAT_nll_surv_a0.0_5foldcv_gc32_concat/tcga_brca/_MCAT_nll_surv_a0.0_5foldcv_gc32_concat_s1/s_0_checkpoint.pt', weights_only=True))
model = model.to(device)
model.eval()

# ResNet on CPU - this is the key change
cpu_device = 'cpu'
resnet = torchvision.models.resnet50(pretrained=True)
resnet.eval()
resnet = resnet.to(cpu_device)

patient_id = "TCGA-E2-A155"
dataframe = pd.read_csv("/work/ai4bio2024/brca_surv/dataset/dataset_brca_wsi.csv")
patient_row = dataframe.loc[dataframe["case_id"] == patient_id].iloc[0]
id_ = patient_row["id"]
slide_id = patient_row["slide_id"]

wsi_path = f'/work/h2020deciderficarra_shared/TCGA/BRCA/data/{id_}/{slide_id}.svs'
slide = openslide.OpenSlide(wsi_path)

# Get image dimensions
width, height = slide.dimensions

mask_img_path = f'/work/h2020deciderficarra_shared/TCGA/BRCA/patches_CLAM/masks/{slide_id}.jpg'
mask_img = np.array(Image.open(mask_img_path))
height_mask, width_mask = mask_img.shape[:2]

original_dim = (width, height)
new_dim = (width_mask, height_mask)

h5_file_path = f"/work/h2020deciderficarra_shared/TCGA/BRCA/features_UNI/h5_files/{slide_id}.h5"
with h5py.File(h5_file_path, 'r') as f:
    coordinates = f['coords'][:]

uni_file_path = f'/work/h2020deciderficarra_shared/TCGA/BRCA/features_UNI/pt_files/{slide_id}.pt'
data = torch.load(uni_file_path, weights_only=True)
data = data.unsqueeze(0)
(data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c) = val_dataset.__getitem__(141)
with torch.no_grad():
    data = data.to(device)
    data_omic1 = data_omic1.type(torch.FloatTensor).to(device)
    data_omic2 = data_omic2.type(torch.FloatTensor).to(device)
    data_omic3 = data_omic3.type(torch.FloatTensor).to(device)
    data_omic4 = data_omic4.type(torch.FloatTensor).to(device)
    data_omic5 = data_omic5.type(torch.FloatTensor).to(device)
    data_omic6 = data_omic6.type(torch.FloatTensor).to(device)
    hazards, survival, output, attn = model(x_path=data, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, 
                                          x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)

survival = survival.cpu().numpy()
bins = np.array([3.8, 24.0, 46.0, 77.9, 244.9])
mid_bin_times = (bins[:-1] + bins[1:]) / 2

expected_survival = (survival * mid_bin_times).sum()

print(f"Predicted probabilities per bin: {survival}")
print(f"Predicted Overall Survival: {expected_survival:.2f} mesi")

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

patch_dim = round(1024 * width_mask / width)

heatmap = np.zeros((height_mask, width_mask))
alpha = 0.6

# Add try-except block to help debug any issues
try:
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

    img_with_heatmap = Image.fromarray(mask_img)
    img_with_heatmap.save("attention_map_fixed.png")
    print("Heatmap generated and saved successfully!")

except Exception as e:
    print(f"Error occurred during heatmap generation: {e}")