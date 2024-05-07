import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from sklearn.model_selection import train_test_split
from pcntoolkit.normative import estimate, evaluate
from plot_num_subjs import plot_num_subjs
from Utility_Functions_Predict_Age import create_design_matrix_one_gender_predict_age, plot_data_with_spline_one_gender_predict_age
from Utility_Functions_Predict_Age import create_dummy_design_matrix_one_gender_predict_age
from Utility_Functions_Predict_Age import barplot_performance_values, plot_y_v_yhat_one_gender, makenewdir, movefiles
from Utility_Functions_Predict_Age import write_ct_to_file_by_gender
from Load_Genz_Data import load_genz_data

def norm_model_predict_age_from_cortthick(gender, orig_struct_var, show_plots, show_nsubject_plots, spline_order,
                                          spline_knots, orig_data_dir, working_dir):

    # load visit 1 (pre-COVID) data
    visit = 1
    brain_good, all_data, roi_ids = load_genz_data(orig_struct_var, visit, orig_data_dir)

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

    # make directories to store files
    makenewdir('{}/data/'.format(working_dir))
    makenewdir('{}/data/{}'.format(working_dir, struct_var))
    makenewdir('{}/data/{}/plots'.format(working_dir, struct_var))
    makenewdir('{}/data/{}/age_models'.format(working_dir, struct_var))
    makenewdir('{}/data/{}/covariate_files'.format(working_dir, struct_var))
    makenewdir('{}/data/{}/response_files'.format(working_dir, struct_var))

    if gender == 'male':
        # remove subject 525 who has an incidental finding
        brain_good = brain_good[~brain_good['participant_id'].isin([525])]
        all_data = all_data[~all_data['participant_id'].isin([525])]

    # show bar plots with number of subjects per age group in pre-COVID data
    if gender == "female":
        genstring = 'Female'
    elif gender == "male":
        genstring = 'Male'
    if show_nsubject_plots:
        plot_num_subjs(all_data, gender, f'{genstring} Subjects by Age with Pre-COVID Data\n'
                                 '(Total N=' + str(all_data.shape[0]) + ')', struct_var, 'pre-covid_allsubj',
                                  working_dir)

    # read in file of subjects in test set at ages 9, 11 and 13
    fname = '{}/visit1_subjects_excluded_from_normative_model_test_set_{}_9_11_13.txt'.format(orig_data_dir,
                                                                                              orig_struct_var)
    subjects_test = pd.read_csv(fname, header=None)

    # exclude subjects from the training set who are in test set
    brain_good = brain_good[~brain_good['participant_id'].isin(subjects_test[0])]
    all_data = all_data[~all_data['participant_id'].isin(subjects_test[0])]

    # plot number of subjects of each gender by age who are included in training data set
    if show_nsubject_plots:
        plot_num_subjs(all_data, gender, f'{genstring} Subjects by Age with Pre-COVID Data\nUsed to Create Model\n'
                                 '(Total N=' + str(all_data.shape[0]) + ')', struct_var, 'pre-covid_norm_model',
                                  working_dir)

    # drop rows with any missing values
    all_data = all_data.dropna()
    all_data.reset_index(inplace=True, drop=True)

    # separate the brain features (response variables) and predictors (corticalthicknes) in to separate dataframes
    all_data_covariates_orig = all_data.loc[:, roi_ids]
    all_data_features = all_data[['age', 'agedays']]

    # average cortical thickness across all regions for each subject
    all_data_covariates = all_data_covariates_orig.mean(axis=1).to_frame()
    all_data_covariates.rename(columns={0: 'avgcortthick'}, inplace=True)

    #find max and min cortthick
    cortthick_min = all_data_covariates['avgcortthick'].min()
    cortthick_max = all_data_covariates['avgcortthick'].max()

    write_ct_to_file_by_gender(working_dir, cortthick_min, cortthick_max, gender)

    # use entire training set to create models
    X_train = all_data_covariates.copy()
    X_test = all_data_covariates.copy()
    y_train = all_data_features.copy()
    y_test = all_data_features.copy()

    # save the subject numbers for the training and validation sets to variables
    s_index_train = X_train.index.values
    s_index_test = X_test.index.values
    subjects_train = all_data.loc[s_index_train, 'participant_id'].values
    subjects_test = all_data.loc[s_index_test, 'participant_id'].values

    # drop the age column from the train data set because we want to use agedays as a target
    y_train.drop(columns=['age'], inplace=True)
    y_test.drop(columns=['age'], inplace=True)

    # change the indices in the train and validation data sets because nan values were dropped above
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    ##########
    # Set up output directories. Save each brain region to its own text file, organized in separate directories,
    # because for each response variable Y (brain region) we fit a separate normative mode
    ##########
    for c in y_train.columns:
        y_train[c].to_csv(f'{working_dir}/resp_tr_' + c + '.txt', header=False, index=False)
        X_train.to_csv(f'{working_dir}/cov_tr.txt', sep='\t', header=False, index=False)
        y_train.to_csv(f'{working_dir}/resp_tr.txt', sep='\t', header=False, index=False)
    for c in y_test.columns:
        y_test[c].to_csv(f'{working_dir}/resp_te_' + c + '.txt', header=False, index=False)
        X_test.to_csv(f'{working_dir}/cov_te.txt', sep='\t', header=False, index=False)
        y_test.to_csv(f'{working_dir}/resp_te.txt', sep='\t', header=False, index=False)

    for i in ['agedays']:
        agedirname = '{}/data/{}/age_models/{}'.format(working_dir, struct_var, i)
        makenewdir(agedirname)
        resp_tr_filename = "{}/resp_tr_{}.txt".format(working_dir, i)
        resp_tr_filepath = agedirname + '/resp_tr.txt'
        shutil.copyfile(resp_tr_filename, resp_tr_filepath)
        resp_te_filename = "{}/resp_te_{}.txt".format(working_dir, i)
        resp_te_filepath = agedirname + '/resp_te.txt'
        shutil.copyfile(resp_te_filename, resp_te_filepath)
        cov_tr_filepath = agedirname + '/cov_tr.txt'
        shutil.copyfile("{}/cov_tr.txt".format(working_dir), cov_tr_filepath)
        cov_te_filepath = agedirname + '/cov_te.txt'
        shutil.copyfile("{}/cov_te.txt".format(working_dir), cov_te_filepath)

    movefiles("{}/resp_*.txt".format(working_dir), "{}/data/{}/response_files/".format(working_dir, struct_var))
    movefiles("{}/cov_t*.txt".format(working_dir), "{}/data/{}/covariate_files/".format(working_dir, struct_var))

    #  this path is where age_models folders are located
    data_dir = '{}/data/{}/age_models/'.format(working_dir, struct_var)

    # Create Design Matrix and add in spline basis and intercept for validation and training data
    create_design_matrix_one_gender_predict_age('test', cortthick_min, cortthick_max, spline_order, spline_knots,
                                                ['agedays'], data_dir)
    create_design_matrix_one_gender_predict_age('train', cortthick_min, cortthick_max, spline_order, spline_knots,
                                                ['agedays'], data_dir)

    # create dataframe with subject numbers to put the Z scores in. Here 'test' refers to the validation set
    subjects_test = subjects_test.reshape(-1, 1)
    subjects_train = subjects_train.reshape(-1, 1)
    Z_score_test_matrix = pd.DataFrame(subjects_test, columns=['subject_id_test'])
    Z_score_train_matrix = pd.DataFrame(subjects_train, columns=['subject_id_train'])

    # Estimate the normative model using a for loop to iterate over brain regions. The estimate function uses a few
    # specific arguments that are worth commenting on:
    # ●alg=‘blr’: specifies we should use BLR. See Table1 for other available algorithms
    # ●optimizer=‘powell’:usePowell’s derivative-free optimization method(faster in this case than L-BFGS)
    # ●savemodel=True: do not write out the final estimated model to disk
    # ●saveoutput=False: return the outputs directly rather than writing them to disk
    # ●standardize=False: do not standardize the covariates or response variable

    # Loop through ROIs

    for a in ['agedays']:
        age_dir = os.path.join(data_dir, a)
        model_dir = os.path.join(data_dir, a, 'Models')
        os.chdir(age_dir)

        # configure the covariates to use. Change *_bspline_* to *_int_*
        cov_file_tr = os.path.join(age_dir, 'cov_bspline_tr.txt')
        cov_file_te = os.path.join(age_dir, 'cov_bspline_te.txt')

        # load train & test response files
        resp_file_tr = os.path.join(age_dir, 'resp_tr.txt')
        resp_file_te = os.path.join(age_dir, 'resp_te.txt')

        # calculate a model based on the training data and apply to the validation dataset. If the model is being created
        # from the entire training set, the validation set is simply a copy of the full training set and the purpose of
        # running this function is to creat and save the model, not to evaluate performance. The following are calcualted:
        # the predicted validation set response (yhat_te), the variance of the predicted response (s2_te), the model
        # parameters (nm),the Zscores for the validation data, and other various metrics (metrics_te)
        yhat_te, s2_te, nm, Z_te, metrics_te = estimate(cov_file_tr, resp_file_tr, testresp=resp_file_te,
                                                        testcov=cov_file_te, alg='blr', optimizer='powell',
                                                        savemodel=True, saveoutput=False, standardize=False)

        # create dummy design matrices for visualizing model
        dummy_cov_file_path = \
            (create_dummy_design_matrix_one_gender_predict_age(struct_var, cortthick_min, cortthick_max, cov_file_tr, spline_order, spline_knots,
                                                   working_dir))

        # compute splines and superimpose on data. Show on screen or save to file depending on show_plots value.
        plot_data_with_spline_one_gender_predict_age(gender, 'Training Data', struct_var, cov_file_tr, resp_file_tr, dummy_cov_file_path,
                              model_dir, a, show_plots, working_dir)


        # store z score for ROI validation set
        Z_score_test_matrix[a] = Z_te

      # savez scores to file
    Z_score_test_matrix.to_csv('{}/data/{}/Z_scores_validation_set_{}.txt'.format(working_dir, struct_var,
                                            gender), index=False)


    return Z_score_train_matrix