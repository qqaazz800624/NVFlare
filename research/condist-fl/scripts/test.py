#%% 

import os, json
import nibabel as nib
import numpy as np


#%% 

label_dir = "/home/u/qqaazz800624/NVFlare/research/condist-fl/scripts/labels"

datalist_path = "/data2/open_dataset/AMOS22/amos22_ct.json"

with open(datalist_path, "r") as f:
    datalist = json.load(f)

datalist

#%%

for key in datalist.keys():
    print(key, len(datalist[key]))


#%%

label_path = os.path.join(label_dir, "LB_AMOS_0001.nii.gz")
label = nib.load(label_path)
label_data = np.asanyarray(label.dataobj)
label_data

#%%


np.unique(label_data)


#%%




#%%