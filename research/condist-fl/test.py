#%%

import torch


#%%

ckpt_path = "/home/u/qqaazz800624/NVFlare/research/condist-fl/workspace_GA_mednext/simulate_job/app_server/best_FL_global_model.pt"

state_dict = torch.load(ckpt_path, map_location='cpu')

#%%

state_dict.keys()




#%%







#%%