#####
# This program implements the bayesian linear regression normative modeling outlined by Rutherford et al.
# NatureProtocols 2022 (https://doi.org/10.1038/s41596-022-00696-5). Here the modeling is used to take the mean
# cortical thickness from all brain regions and model age. The age estimates are used to calculate brain age
# acceleration
######

import matplotlib.pyplot as plt
import pandas as pd
from norm_model_predict_age_from_cortthick import norm_model_predict_age_from_cortthick
from apply_normative_model_predict_age_from_cortthick import apply_normative_model_predict_age_from_cortthick
from plot_z_scores import plot_and_compute_zcores_by_gender


orig_struct_var = 'cortthick'
show_plots = 0  #set to 1 to show training and test data ymvs yhat and spline fit plots.
show_nsubject_plots = 0 #set to 1 to plot number of subjects used in analysis, for each age and gender
spline_order = 1 # order of spline to use for model
spline_knots = 2 # number of knots in spline to use in model

orig_data_dir = '/home/toddr/neva/PycharmProjects/TestPCNNatureProtTutBinaryGenderCortthick'
working_dir = '/home/toddr/neva/PycharmProjects/Adol_Norm_Model_MFSeparate'

#turn off interactive mode, don't show plots unless plt.show() is specified
plt.ioff()

Z_time1 = {}
Z_time2 = {}

for gender in ['male', 'female']:
    norm_model_predict_age_from_cortthick(gender, orig_struct_var, show_plots, show_nsubject_plots, spline_order,
                                          spline_knots, orig_data_dir, working_dir)

    apply_normative_model_predict_age_from_cortthick(gender, orig_struct_var, show_plots, show_nsubject_plots,
                                                     spline_order, spline_knots, orig_data_dir, working_dir)

Z_time2_male  = pd.read_csv('{}/predict_files/{}/Z_scores_by_region_postcovid_testset_Final.txt'
                            .format(working_dir, 'cortthick_male'))
Z_time2_female  = pd.read_csv('{}/predict_files/{}/Z_scores_by_region_postcovid_testset_Final.txt'
                            .format(working_dir, 'cortthick_female'))

Z_time2_male.to_csv(f'{working_dir}/predict_files/Z_time2_male.csv', index=False)
Z_time2_female.to_csv(f'{working_dir}/predict_files/Z_time2_female.csv', index=False)

Z_time2['male'] = Z_time2_male
Z_time2['female'] = Z_time2_female

plot_and_compute_zcores_by_gender(orig_struct_var, Z_time2)

mystop=1