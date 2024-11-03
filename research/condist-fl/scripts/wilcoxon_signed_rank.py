#%%

import numpy as np
from scipy import stats

story2 = [0.77, 0.49, 0.66, 0.28, 0.38]
story1 = [0.40, 0.72, 0.00, 0.36, 0.55]

differences = np.array(story2) - np.array(story1)

stat, p_value = stats.wilcoxon(story2, story1, alternative='greater')

print("p-value of Wilcoxon signed-rank test", p_value)
print("statistic of Wilcoxon signed-rank test", stat)

alpha = 0.05

if p_value < alpha:
    print("Reject the null hypothesis.")
    print("The story 2 is statistically significantly better than the story 1.")
else:
    print("Cannot reject the null hypothesis.")
    print("The story 2 is not statistically significantly better than the story 1.")

#%%






#%%








#%%