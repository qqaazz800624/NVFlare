#%%

import json
import numpy as np
import pandas as pd

# 定義 JSON 檔案路徑
global_json = "/home/u/qqaazz800624/NVFlare/research/condist-fl/infer/dice_scores_kidney_tumor_global.json"
kidney_json = "/home/u/qqaazz800624/NVFlare/research/condist-fl/infer/dice_scores_kidney_tumor_kidney.json"
liver_json = "/home/u/qqaazz800624/NVFlare/research/condist-fl/infer/dice_scores_kidney_tumor_liver.json"
pancreas_json = "/home/u/qqaazz800624/NVFlare/research/condist-fl/infer/dice_scores_kidney_tumor_pancreas.json"
spleen_json = "/home/u/qqaazz800624/NVFlare/research/condist-fl/infer/dice_scores_kidney_tumor_spleen.json"

global_json_GA = "/home/u/qqaazz800624/NVFlare/research/condist-fl/infer_GA/dice_scores_kidney_tumor_global.json"
kidney_json_GA = "/home/u/qqaazz800624/NVFlare/research/condist-fl/infer_GA/dice_scores_kidney_tumor_kidney.json"
liver_json_GA = "/home/u/qqaazz800624/NVFlare/research/condist-fl/infer_GA/dice_scores_kidney_tumor_liver.json"
pancreas_json_GA = "/home/u/qqaazz800624/NVFlare/research/condist-fl/infer_GA/dice_scores_kidney_tumor_pancreas.json"
spleen_json_GA = "/home/u/qqaazz800624/NVFlare/research/condist-fl/infer_GA/dice_scores_kidney_tumor_spleen.json"

# 讀取 JSON 檔案
with open(global_json, "r") as f:
    dice_scores_global = json.load(f)

with open(kidney_json, "r") as f:
    dice_scores_kidney = json.load(f)

with open(liver_json, "r") as f:
    dice_scores_liver = json.load(f)

with open(pancreas_json, "r") as f:
    dice_scores_pancreas = json.load(f)

with open(spleen_json, "r") as f:
    dice_scores_spleen = json.load(f)

with open(global_json_GA, "r") as f:
    dice_scores_global_GA = json.load(f)

with open(kidney_json_GA, "r") as f:
    dice_scores_kidney_GA = json.load(f)

with open(liver_json_GA, "r") as f:
    dice_scores_liver_GA = json.load(f)

with open(pancreas_json_GA, "r") as f:
    dice_scores_pancreas_GA = json.load(f)

with open(spleen_json_GA, "r") as f:
    dice_scores_spleen_GA = json.load(f)


#%%

# 轉換為 numpy 陣列（確保後續操作的一致性）
dice_scores_global = np.array(dice_scores_global)
dice_scores_kidney = np.array(dice_scores_kidney)
dice_scores_liver = np.array(dice_scores_liver)
dice_scores_pancreas = np.array(dice_scores_pancreas)
dice_scores_spleen = np.array(dice_scores_spleen)

dice_scores_global_GA = np.array(dice_scores_global_GA)
dice_scores_kidney_GA = np.array(dice_scores_kidney_GA)
dice_scores_liver_GA = np.array(dice_scores_liver_GA)
dice_scores_pancreas_GA = np.array(dice_scores_pancreas_GA)
dice_scores_spleen_GA = np.array(dice_scores_spleen_GA)

# 利用 pandas 建立 DataFrame 並新增 model 欄位
df_global = pd.DataFrame({
    'score': dice_scores_global.flatten(),
    'model': 'global',
    'method': 'ConDistFL',
    'backbone': 'nnU-Net'
})

df_kidney = pd.DataFrame({
    'score': dice_scores_kidney.flatten(),  # 若原始資料為多維，使用 flatten() 轉成一維
    'model': 'kidney',
    'method': 'ConDistFL',
    'backbone': 'nnU-Net'
})

df_liver = pd.DataFrame({
    'score': dice_scores_liver.flatten(),
    'model': 'liver',
    'method': 'ConDistFL',
    'backbone': 'nnU-Net'
})

df_pancreas = pd.DataFrame({
    'score': dice_scores_pancreas.flatten(),
    'model': 'pancreas',
    'method': 'ConDistFL',
    'backbone': 'nnU-Net'
})

df_spleen = pd.DataFrame({
    'score': dice_scores_spleen.flatten(),
    'model': 'spleen',
    'method': 'ConDistFL',
    'backbone': 'nnU-Net'
})

df_global_GA = pd.DataFrame({
    'score': dice_scores_global_GA.flatten(),
    'model': 'global',
    'method': 'ConDistFL+GA',
    'backbone': 'nnU-Net'
})

df_kidney_GA = pd.DataFrame({
    'score': dice_scores_kidney_GA.flatten(),
    'model': 'kidney',
    'method': 'ConDistFL+GA',
    'backbone': 'nnU-Net'
})

df_liver_GA = pd.DataFrame({
    'score': dice_scores_liver_GA.flatten(),
    'model': 'liver',
    'method': 'ConDistFL+GA',
    'backbone': 'nnU-Net'
})

df_pancreas_GA = pd.DataFrame({
    'score': dice_scores_pancreas_GA.flatten(),
    'model': 'pancreas',
    'method': 'ConDistFL+GA',
    'backbone': 'nnU-Net'
})

df_spleen_GA = pd.DataFrame({
    'score': dice_scores_spleen_GA.flatten(),
    'model': 'spleen',
    'method': 'ConDistFL+GA',
    'backbone': 'nnU-Net'
})

# 將所有 DataFrame 合併成一個 DataFrame
df_all = pd.concat([df_global, df_kidney, df_liver, df_pancreas, df_spleen, df_global_GA, df_kidney_GA, df_liver_GA, df_pancreas_GA, df_spleen_GA], ignore_index=True)
df_all

output_csv_path = "/home/u/qqaazz800624/NVFlare/research/condist-fl/scripts/dice_scores_kidney_tumor.csv"
df_all.to_csv(output_csv_path, index=False)

print(f"CSV file has been saved to {output_csv_path}.")

#%%








#%%










#%%