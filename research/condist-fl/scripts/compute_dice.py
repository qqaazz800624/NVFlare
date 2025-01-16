from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
import torch
from fire import Fire
from tqdm import tqdm


def compute_dice(
    preds: torch.Tensor,
    targets: torch.Tensor,
    label: int
) -> float:
    preds = torch.where(preds == label, 1, 0).int()
    targets = torch.where(targets == label, 1, 0).int()

    sum_p = preds.sum()
    sum_t = targets.sum()

    if sum_t == 0:
        return 1.0 if sum_p == 0 else 0.0

    gt = torch.sum(preds * targets)
    dice = 2.0 * gt / (sum_p + sum_t)

    return dice.cpu().item()

def main(preds_dir: str, label_dir: str, device: str = "cuda:0"):
    metrics = []

    targets = list(Path(label_dir).glob("LB*.gz"))
    for target in tqdm(targets, desc="Computing Dice score for AMOS dataset"):
        case = target.name.split("_")[2].removesuffix(".nii.gz")
        preds = Path(preds_dir) / f"amos_{case}_seg.nii.gz"

        p_data = nib.load(str(preds)).get_fdata()
        t_data = nib.load(str(target)).get_fdata()

        p_data = torch.from_numpy(p_data).to(device)
        t_data = torch.from_numpy(t_data).to(device)

        # Merge tumor labels to organ
        p_data = torch.where(p_data == 2, 1, p_data)
        p_data = torch.where(p_data == 5, 4, p_data)
        p_data = torch.where(p_data == 7, 6, p_data)

        # Kidney, Liver, Pancreas, Spleen
        row = [case]
        for c in [6, 1, 4, 3]:
            row.append(compute_dice(p_data, t_data, c))
        metrics.append(row)

    df = pd.DataFrame(metrics, columns=["Case", "Kidney", "Liver", "Pancreas", "Spleen"])
    df.to_csv(f"{preds_dir}/metrics.csv", index=False)

if __name__ == "__main__":
    Fire(main)

