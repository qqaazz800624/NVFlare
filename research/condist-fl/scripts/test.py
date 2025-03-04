#%%

import numpy as np

a = [1, 2, 3]
b = a
b[0] = 10

print(a[0])



#%%

import torch
arr = torch.randn(2, 3, 5)
torch.permute(arr, (2, 0, 1)).size()

#%%

torch.hstack((arr, arr))




#%%








#%%








#%%