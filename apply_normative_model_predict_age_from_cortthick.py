#####
# This program imports the model and Z-scores from the bayesian linear regression normative modeling of the
# training data set (which is the adolescent visit 1 data). It then uses the model to calculate Z-scores for
# the post-covid adolescent (visit 2) data.
# Author: Neva M. Corrigan
# Date: 21 February, 2024
######
import os
import pandas as pd
from matplotlib import pyplot as plt
from Load_Genz_Data import load_genz_data
from plot_num_subjs import plot_num_subjs
from Utility_Functions_Predict_Age import makenewdir, movefiles, create_dummy_design_matrix_one_gender_predict_age
from Utility_Functions_Predict_Age import plot_data_with_spline_one_gender_predict_age
from Utility_Functions_Predict_Age import create_design_matrix_one_gender_predict_age, read_ct_from_file
import shutil
from normative_edited import predict

def apply_normative_model_predict_age_from_cortthick(gender, orig_struct_var, show_plots, show_nsubject_plots, spline_order, spline_knots,
                                orig_data_dir, working_dir):

    ######################## Apply Normative Model to Post-Covid Data ############################

    # load all brain and behavior data for visit 2
    visit = 2
    brain_good, all_data, roi_ids = load_genz_data(orig_struct_var, visit, orig_data_dir)

    #load brain and behavior data for visit 1
    visit = 1
    brain_v1, all_v1, roi_v1 = load_genz_data(orig_struct_var, visit, orig_data_dir)

    #extract subject numbers from visit 1 and find subjects in visit 2 that aren't in visit 1
    subjects_visit1 = all_v1['participant_id']
    rows_in_v2_but_not_v1 = all_data[~all_data['participant_id'].isin(all_v1['participant_id'])].dropna()
    subjs_in_v2_not_v1 = rows_in_v2_but_not_v1['participant_id'].copy()
    subjs_in_v2_not_v1 = subjs_in_v2_not_v1.astype(int)
    #only keep subjects at 12, 14 and 16 years of age (subject numbers <400) because cannot model 18 and 20 year olds
    subjs_in_v2_not_v1 = subjs_in_v2_not_v1[subjs_in_v2_not_v1 < 400]

    if gender == 'male':
        # keep only data for males
        all_data = all_data.loc[all_data['sex'] == 1]
        struct_var = 'age_male'
    else:
        # keep only data for females
        all_data = all_data.loc[all_data['sex'] == 2]
        struct_var = 'age_female'

    #remove sex column
    all_data = all_data.drop(columns=['sex'])

    #only include subjects that were not in the training set
    fname='{}/visit1_subjects_excluded_from_normative_model_test_set_{}_9_11_13.txt'.format(orig_data_dir, orig_struct_var)
    subjects_to_include = pd.read_csv(fname, header=None)
    subjects_to_include = pd.concat([subjects_to_include, subjs_in_v2_not_v1])
    brain_good = brain_good[brain_good['participant_id'].isin(subjects_to_include[0])]
    all_data = all_data[all_data['participant_id'].isin(subjects_to_include[0])]

    #make file diretories for output
    makenewdir('{}/predict_files/'.format(working_dir))
    makenewdir('{}/predict_files/{}'.format(working_dir, struct_var))
    makenewdir('{}/predict_files/{}/plots'.format(working_dir, struct_var))
    makenewdir('{}/predict_files/{}/age_models'.format(working_dir, struct_var))
    makenewdir('{}/predict_files/{}/covariate_files'.format(working_dir, struct_var))
    makenewdir('{}/predict_files/{}/response_files'.format(working_dir, struct_var))

    # reset indices
    brain_good.reset_index(inplace=True)
    all_data.reset_index(inplace=True, drop=True)
    #read agemin and agemax from file
    cortthickmin, cortthickmax = read_ct_from_file(working_dir, gender)

    #show number of subjects by gender and age
    if gender == "female":
        genstring = 'Female'
    elif gender == "male":
        genstring = 'Male'
    if show_nsubject_plots:
        plot_num_subjs(all_data, gender, f'{genstring} Subjects with Post-COVID Data\nEvaluated by Model\n'
                       +' (Total N=' + str(all_data.shape[0]) + ')', struct_var, 'post-covid_allsubj', working_dir)

    #calculate average cortical thickness across all regions and replace cortthick values in all_data with average
    reg_columns = all_data.columns.to_list()
    sub_info_cols = ['participant_id', 'age', 'agedays']
    reg_columns = [r for r in reg_columns if r not in sub_info_cols]
    all_data_regs = all_data[reg_columns].copy()
    avg_cortthick = all_data_regs.mean(axis=1).to_frame()
    all_data['avgcortthick'] = avg_cortthick
    all_data.drop(columns=reg_columns, inplace=True)

    #specify which columns of dataframe to use as covariates
    X_test = pd.DataFrame(all_data[['avgcortthick']])

    #make a matrix of response variables, one for each brain region
    y_test = pd.DataFrame(all_data.loc[:, 'agedays'])

    #specify paths
    training_dir = '{}/data/{}/age_models/'.format(working_dir, struct_var)
    out_dir = '{}/predict_files/{}/age_models/'.format(working_dir, struct_var)
    #  this path is where ROI_models folders are located
    predict_files_dir = '{}/predict_files/{}/age_models/'.format(working_dir, struct_var)

    ##########
    # Create output directories for each region and place covariate and response files for that region in  each directory
    ##########
    for c in y_test.columns:
        y_test[c].to_csv(f'{working_dir}/resp_te_'+c+'.txt', header=False, index=False)
        X_test.to_csv(f'{working_dir}/cov_te.txt', sep='\t', header=False, index=False)
        y_test.to_csv(f'{working_dir}/resp_te.txt', sep='\t', header=False, index=False)

    for i in ['agedays']:
        roidirname = '{}/predict_files/{}/age_models/{}'.format(working_dir, struct_var, i)
        makenewdir(roidirname)
        resp_te_filename = "{}/resp_te_{}.txt".format(working_dir, i)
        resp_te_filepath = roidirname + '/resp_te.txt'
        shutil.copyfile(resp_te_filename, resp_te_filepath)
        cov_te_filepath = roidirname + '/cov_te.txt'
        shutil.copyfile("{}/cov_te.txt".format(working_dir), cov_te_filepath)

    movefiles("{}/resp_*.txt".format(working_dir), "{}/predict_files/{}/response_files/"
              .format(working_dir, struct_var))
    movefiles("{}/cov_t*.txt".format(working_dir), "{}/predict_files/{}/covariate_files/"
              .format(working_dir, struct_var))

    # # Create Design Matrix and add in spline basis and intercept
    # create_design_matrix_one_gender('test', agemin, agemax, spline_order, spline_knots, roi_ids, out_dir)

    # Create dataframe to store Zscores
    Z_time2 = pd.DataFrame()
    Z_time2['participant_id'] = all_data['participant_id'].copy()
    Z_time2.reset_index(inplace=True, drop = True)

    ####Make Predictions of Brain Structural Measures Post-Covid based on Pre-Covid Normative Model

    data_dir = '{}/predict_files/{}/age_models/'.format(working_dir, struct_var)
    #create design matrices for all regions and save files in respective directories
    create_design_matrix_one_gender_predict_age('test', cortthickmin, cortthickmax, spline_order, spline_knots, ['agedays'], data_dir)


    roi_dir = os.path.join(predict_files_dir, 'agedays')
    model_dir = os.path.join(training_dir, 'agedays', 'Models')
    # os.chdir(roi_dir)

    # configure the covariates to use.
    cov_file_te = os.path.join(roi_dir, 'cov_bspline_te.txt')

    # load test response files
    resp_file_te = os.path.join(roi_dir, 'resp_te.txt')

    # make predictions
    yhat_te, s2_te, Z = predict(cov_file_te, respfile=resp_file_te, alg='blr', model_path=model_dir)

    Z_time2['agedays'] = Z

    #create dummy design matrices
    dummy_cov_file_path= \
        create_dummy_design_matrix_one_gender_predict_age(struct_var, cortthickmin,cortthickmax, cov_file_te,
                                                          spline_order, spline_knots, working_dir)

    plot_data_with_spline_one_gender_predict_age(gender, 'Postcovid (Test) Data ', struct_var, cov_file_te, resp_file_te,
                                     dummy_cov_file_path, model_dir, 'agedays', show_plots, working_dir)

    mystop=1

    Z_time2.to_csv('{}/predict_files/{}/Z_scores_postcovid_testset_{}.txt'
                                .format(working_dir, struct_var, gender), index=False)

    yhat_te_df = pd.DataFrame(yhat_te, columns=['predicted_agedays'])
    yhat_te_df['agedays'] = all_data['agedays']
    yhat_te_df['avgcortthick'] = all_data['avgcortthick']
    yhat_te_df.to_csv('/{}/predict_files/{}/age and predicted age postcovid_test_data_{}.csv'
                                .format(working_dir, struct_var, gender), index=False)

    plt.show()

    return Z_time2

