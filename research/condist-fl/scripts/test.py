#%% Step 1: Normality test

import numpy as np
from scipy import stats

data1 = [0.6384, 0.6899, 0.6460, 0.6287]
data2 = [0.6770, 0.6532, 0.4471, 0.6324]

w1, pvalue1 = stats.shapiro(data1)
w2, pvalue2 = stats.shapiro(data2)

print("Data1 normality test p-value:", pvalue1)
print("Data2 normality test p-value:", pvalue2)

#%% Step 2: Equal variance test

w, pvalue = stats.levene(data1, data2)
print("Equal variance test p-value:", pvalue)

#%%
# two-sample t-test (one tailed)
t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False, alternative='greater')

print("t-statistic:", t_stat)
print("p-value:", p_value)

alpha = 0.05
if t_stat > 0 and p_value < alpha:
    print("Reject null hypothesis. Data 1 is significantly greater than data 2.")
else:
    print("Cannot reject null hypothesis. Data 1 is not significantly greater than data 2.")


#%%