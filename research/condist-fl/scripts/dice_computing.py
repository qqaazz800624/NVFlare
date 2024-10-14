#%%

import nibabel as nib
import numpy as np
import torch
from monai.metrics import DiceMetric
import json
import os

# 0: background
# 1: liver
# 2: liver tumor
# 3: spleen
# 4: pancreas
# 5: pancreas tumor
# 6: kidney
# 7: kidney tumor

class ImageDataset(object):
    def __init__(self, data_root: str, data_list: str, data_list_key: str):
        with open(data_list) as f:
            data = json.load(f).get(data_list_key, [])
        self.data = [os.path.join(data_root, d["label"]) for d in data]
        self.index = 0

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __iter__(self):
        return iter(self.data)


def extract_number_from_filename(filename):
    # Get the basename (file name with extension)
    basename = os.path.basename(filename)  # e.g., 'LB_LVR_80.nii.gz' or 'IM_LVR_80_seg.nii.gz'
    
    # Remove the file extensions
    name_without_ext = basename
    while True:
        name, ext = os.path.splitext(name_without_ext)
        if ext:
            name_without_ext = name
        else:
            break  # No more extensions to remove
    
    # Split by underscore
    parts = name_without_ext.split('_')
    
    # Initialize number as None
    number = None
    
    # Iterate over parts to find the number
    for part in parts:
        if part.isdigit():
            number = part
            break  # Stop after finding the first number
    
    return number

def dice_compute(seg_mask, label):
    model_output_data = seg_mask.astype(np.int32)
    ground_truth_data = label.astype(np.int32)
    
    pred_liver_tumor_mask = (model_output_data == target_class).astype(np.int32)
    gt_liver_tumor_mask = (ground_truth_data == target_class).astype(np.int32)
    
    pred_tensor = torch.from_numpy(pred_liver_tumor_mask)
    gt_tensor = torch.from_numpy(gt_liver_tumor_mask)
    
    pred_tensor = pred_tensor.float()
    gt_tensor = gt_tensor.float()
    
    intersection = torch.sum(pred_tensor * gt_tensor)
    dice_score = (2. * intersection) / (torch.sum(pred_tensor) + torch.sum(gt_tensor) + 1e-6)
    dice_score = dice_score.item()
    return dice_score

#%%

data_root = "/neodata/open_dataset/ConDistFL/data/Liver"
data_list = "/neodata/open_dataset/ConDistFL/data/Liver/datalist.json"
data_list_key = "testing"
target_class = 2

dataset = ImageDataset(data_root, data_list, data_list_key)

dice_scores = []

for i in range(len(dataset)):
    number = extract_number_from_filename(dataset[i])
    
    home_dir = "/home/u/qqaazz800624/NVFlare/research/condist-fl/infer_GA_mednext"
    filename = f"IM_LVR_{number}_seg.nii.gz"
    seg_mask_path = os.path.join(home_dir, filename)
    
    label = nib.load(dataset[i])
    label = label.get_fdata()
    
    seg_mask = nib.load(seg_mask_path)
    seg_mask = seg_mask.get_fdata()
    
    dice_score = dice_compute(seg_mask, label)
    
    dice_scores.append(dice_score)

#%%

dice_scores


#%%

number = extract_number_from_filename(dataset[0])

home_dir = "/home/u/qqaazz800624/NVFlare/research/condist-fl/infer_GA_mednext"
filename = f"IM_LVR_{number}_seg.nii.gz"
seg_mask_path = os.path.join(home_dir, filename)

#%%

label = nib.load(dataset[0])
label = label.get_fdata()

seg_mask = nib.load(seg_mask_path)
seg_mask = seg_mask.get_fdata()

model_output_data = seg_mask.astype(np.int32)
ground_truth_data = label.astype(np.int32)

target_class = 2

pred_liver_tumor_mask = (model_output_data == target_class).astype(np.int32)
gt_liver_tumor_mask = (ground_truth_data == target_class).astype(np.int32)

pred_tensor = torch.from_numpy(pred_liver_tumor_mask)
gt_tensor = torch.from_numpy(gt_liver_tumor_mask)

pred_tensor = pred_tensor.float()
gt_tensor = gt_tensor.float()

intersection = torch.sum(pred_tensor * gt_tensor)
dice_score = (2. * intersection) / (torch.sum(pred_tensor) + torch.sum(gt_tensor) + 1e-6)
dice_score = dice_score.item()

print(f"Dice score: {dice_score:.4f}")

#%%

dice_metric = DiceMetric(include_background=False, reduction="mean")

pred_tensor = pred_tensor.unsqueeze(0).unsqueeze(0)
gt_tensor = gt_tensor.unsqueeze(0).unsqueeze(0)

dice_metric(y_pred=pred_tensor, y=gt_tensor)
dice_score = dice_metric.aggregate().item()
dice_metric.reset()

print(f"Dice score: {dice_score:.4f}")


#%%

import json

datalist_path = "/neodata/open_dataset/ConDistFL/data/Liver/datalist.json"

with open(datalist_path, "r") as f:
    datalist = json.load(f)

datalist['testing']

#%%





#%%










#%%








#%%










#%%








#%%