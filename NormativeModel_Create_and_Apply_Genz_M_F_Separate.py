#####
# This program implements the bayesian linear regression normative modeling outlined by Rutherford et al.
# NatureProtocols 2022 (https://doi.org/10.1038/s41596-022-00696-5). Here the modeling is applied to
# adolescent cortica1 thickness data collected at two time points (before and after the COVID lockdowns).
# This program creates models of cortical thickness change between 9 and 17 years of age for our pre-COVID data and
# stores these models to be applied in another script (Apply_Normative_Model_to_Genz_Time2_Final_Subjects.py)
# to the post-COVID data.
# Author: Neva M. Corrigan
# Date: 21 February, 2024
######

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore
from Utility_Functions import plot_age_acceleration
from make_time1_normative_model import make_time1_normative_model
from apply_normative_model_time2 import apply_normative_model_time2
from plot_z_scores import plot_and_compute_zcores_by_gender
from calculate_avg_brain_age_acceleration_one_gender import calculate_avg_brain_age_acceleration_one_gender_make_model
from calculate_avg_brain_age_acceleration_one_gender import calculate_avg_brain_age_acceleration_one_gender_apply_model
from calculate_avg_brain_age_acceleration_bootstrap import calculate_avg_brain_age_acceleration_one_gender_apply_model_bootstrap
import time

orig_struct_var = 'cortthick'
show_plots = 0  #set to 1 to show training and test data ymvs yhat and spline fit plots.
show_nsubject_plots = 0 #set to 1 to plot number of subjects used in analysis, for each age and gender
spline_order = 1 # order of spline to use for model
spline_knots = 2 # number of knots in spline to use in model
nbootstrap = 10
num_permute = 10000
perform_train_test_split_precovid = 0  # flag indicating whether to split training set (pre-covid data) into train and
                                       # validations (test) sets. If this is set to 0, the entire training set is used
                                       # for the model and there is no validation set. Regardless of the value of this
                                       # flag, no post-covid data is used in creating or evaluating the normative model.
run_make_norm_model = 1
run_apply_norm_model = 0
calc_brain_age_acc = 0
calc_mf_age_acc_diff_permute = 0
calc_CI_age_acc_bootstrap = 0

orig_data_dir = '/home/toddr/neva/PycharmProjects/TestPCNNatureProtTutBinaryGenderCortthick'
working_dir = '/home/toddr/neva/PycharmProjects/Adol_Norm_Model_MFSeparate'

ageacc_from_bootstraps = {}

start=time.time()

for gender in ['male', 'female']:

    if run_make_norm_model:
        Z_time1 = {}

        Z_time1[gender] = make_time1_normative_model(gender, orig_struct_var, show_plots, show_nsubject_plots, spline_order, spline_knots,
                               perform_train_test_split_precovid, orig_data_dir, working_dir)

    if run_apply_norm_model:
        Z_time2 = {}

        Z_time2[gender] = apply_normative_model_time2(gender, orig_struct_var, show_plots, show_nsubject_plots, spline_order, spline_knots,
                                orig_data_dir, working_dir)

        Z_time2_male = pd.read_csv('{}/predict_files/{}/Z_scores_by_region_postcovid_testset_Final.txt'
                                   .format(working_dir, 'cortthick_male'))
        Z_time2_female = pd.read_csv('{}/predict_files/{}/Z_scores_by_region_postcovid_testset_Final.txt'
                                     .format(working_dir, 'cortthick_female'))

        Z_time2_male.to_csv(f'{working_dir}/predict_files/Z_time2_male.csv', index=False)
        Z_time2_female.to_csv(f'{working_dir}/predict_files/Z_time2_female.csv', index=False)

        Z_time2['male'] = Z_time2_male
        Z_time2['female'] = Z_time2_female

        plot_and_compute_zcores_by_gender(orig_struct_var, Z_time2)

    if calc_brain_age_acc:
        mean_agediff = {}

        calculate_avg_brain_age_acceleration_one_gender_make_model(gender, orig_struct_var, show_nsubject_plots, show_plots,
                                                                   spline_order, spline_knots, orig_data_dir, working_dir)

        mean_agediff[gender] = calculate_avg_brain_age_acceleration_one_gender_apply_model(gender, orig_struct_var, show_nsubject_plots, show_plots,
                                                               spline_order, spline_knots, orig_data_dir, working_dir, num_permute=0, permute=False, shuffnum=0)

    if calc_mf_age_acc_diff_permute:
        mean_agediff_permuted_dict = {'male': None, 'female': None}
        male = pd.DataFrame(columns=['mean_agediff'])
        female = pd.DataFrame(columns=['mean_agediff'])

        for i_permute in range(num_permute):
            print(f'i_permute = {i_permute}')
            m = pd.DataFrame(columns=['mean_agediff'])
            m.loc[0, 'mean_agediff'] = calculate_avg_brain_age_acceleration_one_gender_apply_model(gender, orig_struct_var,
                                                                                    show_nsubject_plots, show_plots,
                                                                                    spline_order, spline_knots,
                                                                                    orig_data_dir, working_dir, num_permute=num_permute,
                                                                                    permute=True, shuffnum=i_permute)

            mean_agediff_permuted_dict[gender] = pd.concat([mean_agediff_permuted_dict[gender], m], ignore_index=True)
            # Determine percentile of mean_agediff
            mean_age_diff_permuted_female = mean_agediff_permuted_dict['female'].to_numpy()
            mean_age_diff_permuted_male = mean_agediff_permuted_dict['male'].to_numpy()
            sex_age_diff_array = np.squeeze(mean_age_diff_permuted_female - mean_age_diff_permuted_male)
            empirical_gender_diff = mean_agediff['female'] - mean_agediff['male']
            # Find out what percentile value is with respect to arr
            percentile = percentileofscore(sex_age_diff_array, empirical_gender_diff)
            # Print the percentile
            print("The percentile of", empirical_gender_diff, "with respect to the array is:", percentile)
            # Write empirical percentile and permutation results for sex diff array to file
            # append empirical percentile to end of array
            sex_age_diff_array = np.append(sex_age_diff_array, empirical_gender_diff)
            # Save array to text file
            np.savetxt(f'{working_dir}/sex acceleration distribution.txt', sex_age_diff_array)

    if calc_CI_age_acc_bootstrap:

        ageacc_from_bootstraps[gender] = calculate_avg_brain_age_acceleration_one_gender_apply_model_bootstrap(gender, orig_struct_var, show_nsubject_plots, show_plots,
                                                           spline_order, spline_knots, orig_data_dir, working_dir, nbootstrap)
        # Write age acceleration from bootstrapping to file
        with open(f"{working_dir}/ageacceleration_dictionary {nbootstrap} bootstraps.txt", 'w') as f:
            for key, value in ageacc_from_bootstraps.items():
                f.write('%s:%s\n' % (key, value))

        plot_age_acceleration(working_dir, nbootstrap, mean_agediff)

end = time.time()
print(f'Elapsed time is {(end - start)/60.0} minutes')

mystop=1