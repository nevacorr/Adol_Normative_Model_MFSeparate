#########
# This file contains a number of functions utilized in implementing normative modeling
##########
import os
import numpy as np
from matplotlib import pyplot as plt
from pcntoolkit.normative import predict
import pandas as pd
import seaborn as sns
import shutil
import glob
from pcntoolkit.util.utils import create_bspline_basis
from matplotlib.colors import ListedColormap
from scipy import stats

def makenewdir(path):
    isExist = os.path.exists(path)
    if isExist is False:
        os.mkdir(path)
        print('made directory {}'.format(path))

def movefiles(pattern, folder):
    files = glob.glob(pattern)
    for file in files:
        file_name = os.path.basename(file)
        shutil.move(file, folder + file_name)
        print('moved:', file)

def create_design_matrix_one_gender_predict_age(datatype, cortthickmin, cortthickmax, spline_order, spline_knots, ages, data_dir):
    B = create_bspline_basis(cortthickmin, cortthickmax, p=spline_order, nknots=spline_knots)
    for a in ages:
        print('Creating basis expansion for ROI:', a)
        age_dir = os.path.join(data_dir, a)
        # os.chdir(roi_dir)
        # create output dir
        os.makedirs(os.path.join(age_dir, 'blr'), exist_ok=True)

        # load train & test covariate data matrices
        if datatype == 'train':
            X = np.loadtxt(os.path.join(age_dir, 'cov_tr.txt'))
        elif datatype == 'test':
            X = np.loadtxt(os.path.join(age_dir, 'cov_te.txt'))

        # add intercept column
        X = np.vstack((X, np.ones(len(X)))).T

        if datatype == 'train':
            np.savetxt(os.path.join(age_dir, 'cov_int_tr.txt'), X)
        elif datatype == 'test':
            np.savetxt(os.path.join(age_dir, 'cov_int_te.txt'), X)

        # create Bspline basis set
        # This creates a numpy array called Phi by applying function B to each element of the first column of X_tr
        Phi = np.array([B(i) for i in X[:, 0]])
        X = np.concatenate((X, Phi), axis=1)
        if datatype == 'train':
            np.savetxt(os.path.join(age_dir, 'cov_bspline_tr.txt'), X)
        elif datatype == 'test':
            np.savetxt(os.path.join(age_dir, 'cov_bspline_te.txt'), X)

#this function creates a dummy design matrix for plotting of spline function
def create_dummy_design_matrix_one_gender_predict_age(struct_var, ctmin, ctmax, cov_file, spline_order, spline_knots, path):

    # make dummy test data covariate file starting with a column for age
    dummy_cov = np.linspace(ctmin, ctmax, num=1000)
    ones = np.ones((1, dummy_cov.shape[0]))

    #add a column for intercept
    dummy_cov_final = np.vstack((dummy_cov, ones)).T

    # create spline features and add them to predictor dataframe
    BAll = create_bspline_basis(ctmin, ctmax, p=spline_order, nknots=spline_knots)
    Phidummy = np.array([BAll(i) for i in dummy_cov_final[:, 0]])
    dummy_cov_final = np.concatenate((dummy_cov_final, Phidummy), axis=1)

    # write these new created predictor variables with spline and response variable to file
    dummy_cov_file_path = os.path.join(path, 'cov_file_dummy.txt')
    np.savetxt(dummy_cov_file_path, dummy_cov_final)
    return dummy_cov_file_path


# this function plots  data with spline model superimposed, for both male and females
def plot_data_with_spline_one_gender_predict_age(gender, datastr, struct_var, cov_file, resp_file, dummy_cov_file_path, model_dir, age,
                                     showplots, working_dir):

    output = predict(dummy_cov_file_path, respfile=None, alg='blr', model_path=model_dir)

    yhat_predict_dummy=output[0]

    # load real data predictor variables for region
    X = np.loadtxt(cov_file)
    # load real data response variables for region
    y = np.loadtxt(resp_file)

    # create dataframes for plotting with seaborn facetgrid objects
    dummy_cov = np.loadtxt(dummy_cov_file_path)
    df_origdata = pd.DataFrame(data=X[:, 0], columns=['avgcortthick'])
    df_origdata[struct_var] = y.tolist()
    df_origdata[struct_var] = df_origdata[struct_var]/365.25
    df_estspline = pd.DataFrame(data=dummy_cov[:, 0].tolist(),columns=['avgcortthick'])
    tmp = np.array(yhat_predict_dummy.tolist(), dtype=float)
    df_estspline[struct_var] = tmp
    df_estspline[struct_var] = df_estspline[struct_var]/365.25
    df_estspline = df_estspline.drop(index=df_estspline.iloc[999].name).reset_index(drop=True)

    fig=plt.figure()
    if gender == 'female':
        color = 'green'
    else:
        color = 'blue'
    sns.lineplot(data=df_estspline, x='avgcortthick', y=struct_var, color=color, legend=False)
    sns.scatterplot(data=df_origdata, x='avgcortthick', y=struct_var, color=color)
    ax = plt.gca()
    fig.subplots_adjust(right=0.82)
    plt.title(datastr +' ' + struct_var +  ' vs. avgcortthick\n')
    plt.xlabel('AvgCortthick')
    plt.ylabel(datastr + ' ' +struct_var)
    plt.show()
    plt.savefig('{}/data/{}/plots/{}_vs_age_withsplinefit_{}_{}'
                .format(working_dir, struct_var, struct_var, age, datastr))
    plt.close(fig)
    #write data to file if training set so male and female data and models can be viewed on same plot
    if datastr == 'Training Data':
        splinemodel_fname = f'{working_dir}/data/{struct_var}/plots/spline_model_{datastr}_{age}_{gender}.csv'
        origdata_fname = f'{working_dir}/data/{struct_var}/plots/datapoints_{datastr}_{age}_{gender}.csv'
        df_estspline.to_csv(splinemodel_fname)
        df_origdata.to_csv(origdata_fname)


def plot_y_v_yhat_one_gender(gender, cov_file, resp_file, yhat, typestring, struct_var, roi, Rho, EV):
    cov_data = np.loadtxt(cov_file)
    y = np.loadtxt(resp_file).reshape(-1,1)
    dfp = pd.DataFrame()
    y=y.flatten()
    yht=yhat.flatten()
    dfp['y'] = y
    dfp['yhat'] = yhat
    print(dfp.dtypes)
    fig = plt.figure()
    if gender == 'female':
        color='green'
    else:
        color='blue'

    sns.scatterplot(data=dfp, x='y', y='yhat', color=color)
    ax = plt.gca()
    fig.subplots_adjust(right=0.82)
    plt.title(typestring + ' ' + struct_var + ' vs. estimate\n'
              + roi +' EV=' + '{:.4}'.format(str(EV.item())) + ' Rho=' + '{:.4}'.format(str(Rho.item())))
    plt.xlabel(typestring + ' ' + struct_var)
    plt.ylabel(struct_var + ' estimate on ' + typestring)
    plt.plot([min(y), max(y)], [min(y), max(y)], color='red')  # plots line y = x
    plt.show(block=False)

def barplot_performance_values(struct_var, metric, df, spline_order, spline_knots, datastr, path, gender):
    colors = ['blue' if 'lh' in x else 'green' for x in df.ROI]
    fig, ax = plt.subplots(figsize=(8, 10))
    sns.barplot(x=df[metric], y=df['ROI'], hue=df['ROI'], orient='h', palette=colors, legend=False)
    plt.subplots_adjust(left=0.4)
    plt.subplots_adjust(top=0.93)
    plt.subplots_adjust(bottom=0.05)
    ax.set_title('Test Set ' + metric + ' for All Brain Regions' + ' ' + gender)
    plt.show(block=False)
    plt.savefig(
        '{}/data/{}/plots/{}_{}_for_all_regions_splineorder{}, splineknots{}_{}.png'
        .format(path, struct_var, datastr, metric, spline_order, spline_knots, gender))
    #plt.close(fig)

def write_ct_to_file_by_gender(wdir, cortthickmin, cortthickmax, gender):
    with open("{}/cortthickmin_cortthickmax_Xtrain_{}.txt".format(wdir, gender), "w") as file:
        # Write the values to the file
        file.write(f'{cortthickmin}\n')
        file.write(f'{cortthickmax}\n')

def read_ct_from_file(wdir,gender):  #note this didn't have  gender before? Check code for norm model of CT
    # Open the file in read mode
    with open("{}/cortthickmin_cortthickmax_Xtrain_{}.txt".format(wdir, gender), "r") as file:
        # Read all lines from the file
        lines = file.readlines()
    # Extract the values from the lines
    cortthickmin = float(lines[0].strip())
    cortthickmax = float(lines[1].strip())
    return cortthickmin, cortthickmax

def write_list_to_file(mylist, filepath):
   with open(filepath, 'w') as file:
       for item in mylist:
           file.write(item + '\n')

def plot_brain_age_gap_by_gender(brain_age_gap_df, model_type, include_gender):
    #input dataframe must have a 'gender' columns and an 'agediff' column
    #plot figure
    fig=plt.figure(figsize=(7,7))
    if include_gender ==0:
        #plot histograms
        mean_diff = brain_age_gap_df['agediff'].mean()
        sns.histplot(data=brain_age_gap_df, x='agediff')
        plt.title(
            f'{model_type} Distributions of difference between post-COVID\n lockdown age and actual age\n '
            f'mean diff = {mean_diff:.1f} years')
    elif include_gender == 1:
       # find mean of age_diff by gender
        means_by_gender = brain_age_gap_df.groupby('gender')['agediff'].mean()
        mean_diff_male = means_by_gender[1]
        mean_diff_female = means_by_gender[2]
        p = {'male': 'blue', 'female': 'orange'}
        brain_age_gap_df['gender'].replace({1: 'male', 2: 'female'}, inplace=True)
        #plot histograms
        sns.histplot(data=brain_age_gap_df, x='agediff', hue='gender', palette = p, element='step')
        plt.title(
            f'{model_type} Distributions of difference between post-COVID\n lockdown age and actual age by gender\n '
            f'mean diff male = {mean_diff_male:.1f} years mean diff female = {mean_diff_female:.1f} years')
    plt.xlabel('Predicted post-Covid lock down age  - actual age (years)')
    plt.ylabel('Number of subjects')
    plt.show(block=False)




