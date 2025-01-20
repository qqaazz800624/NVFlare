import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm


if __name__ == "__main__":
    mapping = {
        1: 3,   # Spleen
        2: 6,   # Kidney
        3: 6,   # Kidney
        6: 1,   # Liver 
        10: 4   # Pancreas
    }
    for i in range(30):
        mapping.setdefault(i, 0)

    with open("amos22.json", "r") as f:
        dataset = json.load(f)

    dataroot = Path("/data2/open_dataset/AMOS22")
    datalist = {}
    for split in ["training", "validation"]:
        datalist[split] = []
        for case in tqdm(dataset[split]):
            seg_name = Path(case["label"]).name.replace("amos", "LB_AMOS")

            # Convert label format before upload
            seg_nifti = nib.load(str(dataroot / case["label"]))
            seg_data = np.asanyarray(seg_nifti.dataobj)
            seg_data = np.vectorize(mapping.get)(seg_data)
            seg_nifti = nib.Nifti1Image(seg_data.astype(np.uint8), seg_nifti.affine)
            nib.save(seg_nifti, f"labels/{seg_name}")
