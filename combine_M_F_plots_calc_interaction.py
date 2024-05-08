#####
# This program reads in the training data spline models for males and females and plots them
# along with the data used to create the models. It also tests for interactions between age and gender.
######
from matplotlib import pyplot as plt
import pandas as pd
from helper_functions_mfinteraction import save_csvdata_to_dictdf, calc_interaction, plot_data

# Read in data files and save to data as dataframe variables
data_dir = '/home/toddr/neva/PycharmProjects/Adol_COVID_cortthick_CentileBrain/data'
spline_model_data = save_csvdata_to_dictdf('spline_model', data_dir)
datapoints_data = save_csvdata_to_dictdf('datapoints', data_dir)

# Save male and female data to separate dataframes
spline_model_female = spline_model_data['female']
spline_model_male = spline_model_data['male']
datapoints_female = datapoints_data['female']
datapoints_male = datapoints_data['male']

# Add gender column to dataframe
spline_model_female['gender'] = 0
spline_model_male['gender'] = 1
datapoints_female['gender'] = 0
datapoints_male['gender'] = 1

# Combine male and female data into one dataframe
spline_models_all = pd.concat([spline_model_female, spline_model_male], axis=0)
datapoints_all = pd.concat([datapoints_female, datapoints_male], axis=0)

# Make variables names so they will be compatible with statsmodels
spline_models_all.rename(columns={'Age in Days': 'Age_in_Days'}, inplace=True)
datapoints_all.rename(columns={'Age in Days': 'Age_in_Days'}, inplace=True)
spline_models_all.columns = spline_models_all.columns.str.replace('-', '_')
datapoints_all.columns = datapoints_all.columns.str.replace('-', '_')

# Make a list of all regions with cortthickness values
regions = datapoints_all.columns.tolist()
regions.remove('Age_in_Days')
regions.remove('gender')

# Calculate statistical significance of interaction between age and gender
interaction_pvals_df, corrected_int_pvals_df = calc_interaction(datapoints_all, regions)

# Make dataframes that only contain regions that shown significant pvalues for interation between age and gender
sig_interaction_uncorr_pvals = interaction_pvals_df.loc[interaction_pvals_df[0]<0.050]
sig_interaction_corr_pvals = corrected_int_pvals_df.loc[corrected_int_pvals_df[0]<0.050]

# Save plots to file and display plots if desired
show_plots=1
plot_data(regions, spline_models_all, datapoints_all, interaction_pvals_df, corrected_int_pvals_df, data_dir, show_plots)

plt.show()
mystop=1
