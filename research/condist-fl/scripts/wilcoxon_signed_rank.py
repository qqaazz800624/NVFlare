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

import json 
import numpy as np

with open("/home/u/qqaazz800624/NVFlare/research/condist-fl/infer_GA/dice_scores.json", "r") as f:
    dice_scores_GA = json.load(f)

with open("/home/u/qqaazz800624/NVFlare/research/condist-fl/infer/dice_scores.json", "r") as f:
    dice_scores = json.load(f)



#%%

dice_scores_dict = {}

dice_scores_dict["GA"] = dice_scores_GA
dice_scores_dict["Non-GA"] = dice_scores

print(dice_scores_dict["GA"])
print(dice_scores_dict["Non-GA"])

with open("/home/u/qqaazz800624/NVFlare/research/condist-fl/scripts/dice_scores_dict.json", "w") as f:
    json.dump(dice_scores_dict, f)


#%%