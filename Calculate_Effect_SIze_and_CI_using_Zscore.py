####
# This program calculate the mean Z score and confidence intervals for males and females. This is one estimate
# of effect size.
####

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

working_dir = '/home/toddr/neva/PycharmProjects/Adol_Norm_Model_MFSeparate'

# Specify filenames for post-covid z-scores
Z_female_time2_file = f'{working_dir}/predict_files/Z_time2_female.csv'
Z_male_time2_file = f'{working_dir}/predict_files/Z_time2_male.csv'

# Load Z scores from post-covid data for males and females
Z2_female = pd.read_csv(Z_female_time2_file)
Z2_male = pd.read_csv(Z_male_time2_file)

# Create list of brain regions
rois = Z2_female.columns.values.tolist()
rois.remove('participant_id')

# Calculate standard deviations for all brain regions for males and females
Z2_stats = pd.DataFrame(index=['mean_female', 'mean_male', 'std_female', 'std_male'])

for col in rois:
    Z2_stats.loc['mean_female', col] = np.mean(Z2_female.loc[:,col])
    Z2_stats.loc['mean_male', col] = np.mean(Z2_male.loc[:,col])
    Z2_stats.loc['std_female', col] = np.std(Z2_female.loc[:,col])
    Z2_stats.loc['std_male', col] = np.std(Z2_male.loc[:,col])

Z2_stats.loc['upper_CI_female',:] = (
        Z2_stats.loc['mean_female',:] + 2 * Z2_stats.loc['std_female'] / math.sqrt(Z2_female.shape[0] - 1))
Z2_stats.loc['lower_CI_female',:] = (
        Z2_stats.loc['mean_female',:] - 2 * Z2_stats.loc['std_female'] / math.sqrt(Z2_female.shape[0] - 1))
Z2_stats.loc['upper_CI_male',:] = (
        Z2_stats.loc['mean_male',:] + 2 * Z2_stats.loc['std_male'] / math.sqrt(Z2_male.shape[0] - 1))
Z2_stats.loc['lower_CI_male',:] = (
        Z2_stats.loc['mean_male',:] - 2 * Z2_stats.loc['std_male'] / math.sqrt(Z2_male.shape[0] - 1))

# Remove prefix from column names
Z2_stats.columns = Z2_stats.columns.str.replace('cortthick-', '')

# Extract mean values and confidence intervals
mean_female = Z2_stats.loc['mean_female']
mean_male = Z2_stats.loc['mean_male']
upper_ci_female = Z2_stats.loc['upper_CI_female']
lower_ci_female = Z2_stats.loc['lower_CI_female']
upper_ci_male = Z2_stats.loc['upper_CI_male']
lower_ci_male = Z2_stats.loc['lower_CI_male']

# Plotting
plt.figure(figsize=(12, 8))

# Plotting mean values with error bars for females
plt.errorbar(x=range(len(mean_female)), y=mean_female, yerr=[mean_female - lower_ci_female,
                                                upper_ci_female - mean_female], fmt='o', label='Female', color='green', markersize=3)

# Plotting mean values with error bars for males
xval = range(len(mean_male))
xval_offset = [p - 0.25 for p in xval ]
plt.errorbar(x=xval_offset, y=mean_male, yerr=[mean_male - lower_ci_male,
                                                upper_ci_male - mean_male], fmt='o', label='Male', color='blue', markersize=3)

plt.xlabel('Brain Region')
plt.ylabel('Mean Value')
plt.title('Mean Zscore by Brain Region with Confidence Intervals')
plt.legend(loc='lower right')

plt.xticks(range(len(mean_female)), mean_female.index, rotation=90)
plt.xlim(-0.8, len(mean_female) - 0.5)
plt.tight_layout()
plt.show()

