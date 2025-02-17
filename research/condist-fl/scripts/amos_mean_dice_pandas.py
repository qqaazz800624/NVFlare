#%% 

import pandas as pd
import os
home_dir = "/home/u/qqaazz800624/NVFlare/research/condist-fl/scripts"
file_path = os.path.join(home_dir, "AMOS22_Results/mednext_kidney.csv")
df = pd.read_csv(file_path)

df.describe()

#%%


