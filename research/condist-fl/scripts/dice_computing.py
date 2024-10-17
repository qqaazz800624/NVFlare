#%%

import nibabel as nib
import numpy as np
import torch
from monai.metrics import DiceMetric
import json
import os
from tqdm import tqdm

label_class_dict = {
    'background': 0,
    'liver': 1,
    'liver_tumor': 2,
    'spleen': 3,
    'pancreas': 4,
    'pancreas_tumor': 5,
    'kidney': 6,
    'kidney_tumor': 7
}

data_prefix_dict = {
    'liver': 'LVR',
    'liver_tumor': 'LVR',
    'spleen': 'SPL',
    'pancreas': 'PAN',
    'pancreas_tumor': 'PAN',
    'kidney': 'KITS',
    'kidney_tumor': 'KITS'
}

data_dir_dict = {
    'liver': 'Liver',
    'liver_tumor': 'Liver',
    'spleen': 'Spleen',
    'pancreas': 'Pancreas',
    'pancreas_tumor': 'Pancreas',
    'kidney': 'KiTS19',
    'kidney_tumor': 'KiTS19'
}

class LabelDataset(object):
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
    
    pred_mask = (model_output_data == target_class).astype(np.int32)
    gt_mask = (ground_truth_data == target_class).astype(np.int32)
    
    pred_tensor = torch.from_numpy(pred_mask)
    gt_tensor = torch.from_numpy(gt_mask)
    
    pred_tensor = pred_tensor.float()
    gt_tensor = gt_tensor.float()
    
    intersection = torch.sum(pred_tensor * gt_tensor)
    dice_score = (2. * intersection) / (torch.sum(pred_tensor) + torch.sum(gt_tensor) + 1e-6)
    dice_score = dice_score.item()
    return dice_score

#%%
target = 'spleen'

data_dir = data_dir_dict[target]
data_root = f"/neodata/open_dataset/ConDistFL/data/{data_dir}"
data_list = f"/neodata/open_dataset/ConDistFL/data/{data_dir}/datalist.json"

methods = ['GA', 'ConDist']

for method in methods:
    
    if method == 'GA':
        home_dir = "/home/u/qqaazz800624/NVFlare/research/condist-fl/infer_GA"
    else:
        home_dir = "/home/u/qqaazz800624/NVFlare/research/condist-fl/infer"

    target_class = label_class_dict[target]
    data_prefix = data_prefix_dict[target]

    dataset = LabelDataset(data_root, data_list, "testing")

    dice_scores = []

    for data in tqdm(dataset):
        number = extract_number_from_filename(data)
        filename = f"IM_{data_prefix}_{number}_seg.nii.gz"
        seg_mask_path = os.path.join(home_dir, filename)
        
        label = nib.load(data)
        label = label.get_fdata()
        
        seg_mask = nib.load(seg_mask_path)
        seg_mask = seg_mask.get_fdata()
        
        dice_score = dice_compute(seg_mask, label)
        
        dice_scores.append(dice_score)

    save_dir = os.path.join(home_dir, "dice_scores.json")

    with open(save_dir, "w") as f:
        json.dump(dice_scores, f)

    print(f"Finished computing dice scores for {method} method.")
#%%

import json 
import numpy as np

with open("/home/u/qqaazz800624/NVFlare/research/condist-fl/infer_GA/dice_scores.json", "r") as f:
    dice_scores_GA = json.load(f)

with open("/home/u/qqaazz800624/NVFlare/research/condist-fl/infer/dice_scores.json", "r") as f:
    dice_scores = json.load(f)


print("The mean of GA method:", np.round(np.mean(dice_scores_GA), 4))
print("The mean of Non-GA method:", np.round(np.mean(dice_scores), 4))

#%%

import numpy as np
from scipy import stats

differences = np.array(dice_scores_GA) - np.array(dice_scores)

# Check the normality with Shapiro-Wilk test
w, p_value = stats.shapiro(differences)
print("p-value of Shapiro-Wilk test", p_value)

# check the normality
alpha = 0.05
if p_value > alpha:
    print("Cannot reject the null hypothesis (normality is not rejected).")
else:
    print("Reject the null hypothesis (normality is rejected).")

#%%

# Wilcoxon signed-rank test

stat, p_value = stats.wilcoxon(dice_scores_GA, dice_scores, alternative='greater')

print("p-value of Wilcoxon signed-rank test", p_value)
print("statistic of Wilcoxon signed-rank test", stat)

alpha = 0.05

if p_value < alpha:
    print("Reject the null hypothesis.")
    print("The GA method is statistically significantly better than the Non-GA method.")
else:
    print("Cannot reject the null hypothesis.")
    print("The GA method is not statistically significantly better than the Non-GA method.")

#%%








#%%










#%%








#%%