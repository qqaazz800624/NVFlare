#%%
import json
import numpy as np
import pandas as pd
import os

def load_scores(filepath):
    """Load JSON data from the given filepath and return a flattened NumPy array."""
    with open(filepath, "r") as f:
        return np.array(json.load(f)).flatten()

# Base directory for your experiment results
home_dir = "/home/u/qqaazz800624/NVFlare/research/condist-fl"

# Inference directories for non-GA and GA results
infer_dirs = {
    "ConDistFL": "infer",      # non-GA
    "ConDistFL+GA": "infer_GA"   # GA
}

# Define the tasks and models.
# For example, here we assume filenames follow: dice_scores_{task}_{model}.json
# You can extend tasks as needed.
tasks = ["kidney", "kidney_tumor", "liver", "liver_tumor", "pancreas", "pancreas_tumor", "spleen"]  # Change or add tasks like "kidney", "liver", "liver_tumor", etc.
models = ["global", "kidney", "liver", "pancreas", "spleen"]

# Build a list of file metadata dictionaries using nested loops.
files = []
for method, subdir in infer_dirs.items():
    for task in tasks:
        for model in models:
            filename = f"dice_scores_{task}_{model}.json"
            filepath = os.path.join(home_dir, subdir, filename)
            files.append({
                "filepath": filepath,
                "task": task,
                "model": model,
                "method": method,
                "backbone": "nnU-Net"
            })

# Create a list of DataFrames from the file metadata.
df_list = []
for info in files:
    if os.path.exists(info["filepath"]):
        df = pd.DataFrame({
            "score": load_scores(info["filepath"]),
            "task": info["task"],
            "model": info["model"],
            "method": info["method"],
            "backbone": info["backbone"]
        })
        df_list.append(df)
    else:
        print(f"Warning: {info['filepath']} does not exist.")

# Concatenate all individual DataFrames into one
if df_list:
    df_all = pd.concat(df_list, ignore_index=True)
else:
    raise ValueError("No valid JSON files found; please check your file paths.")

# Save the final DataFrame as a CSV file.
output_csv_path = os.path.join(home_dir, "scripts", "dice_scores_nnUNet_across_tasks.csv")
df_all.to_csv(output_csv_path, index=False)

print(f"CSV file has been saved to {output_csv_path}.")



#%%










#%%