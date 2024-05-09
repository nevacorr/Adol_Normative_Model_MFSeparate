#####
# Author: Neva M. Corrigan
# Date: 21 February, 2024
######

import numpy as np
import os
from Utility_Functions import fit_regression_model_dummy_data

days_to_years_factor=365.25

def calculate_age_acceleration_one_gender(gender, struct_var, roi_dir, yhat, model_dir, roi,
                               dummy_cov_file_path):

    #load age and gender (predictors)
    actual_age = np.loadtxt(os.path.join(roi_dir, 'cov_te.txt'))
    #load measured struct_var
    actual_struct = np.loadtxt(os.path.join(roi_dir, 'resp_te.txt'))
    predicted_struct = yhat

    slope, intercept= fit_regression_model_dummy_data(model_dir, dummy_cov_file_path)

    #for every female subject, calculate predicted age
    predicted_age = (actual_struct - intercept)/slope

    avg_actual_str = np.mean(actual_struct)

    avg_predicted_age = np.mean(predicted_age)/days_to_years_factor

    avg_actual_age = np.mean(actual_age)/days_to_years_factor

    #subtract mean average age from mean predicted age for each age group
    mean_agediff_f = np.mean(np.subtract(predicted_age, actual_age))/days_to_years_factor

    return mean_agediff_f


