
import glob
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.multitest as smt
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

def save_csvdata_to_dictdf(fileprefix, data_dir):
    data_dict={}
    for sex in ['male', 'female']:
        # Create list of all filenames with cortical thickness data for each region and gender
        spline_model_files = glob.glob(f'{data_dir}/cortthick_{sex}/plots/{fileprefix}*.csv')
        # Create dictionary of dataframes
        data_dict[sex] = pd.DataFrame()
        # Read the data in and save to a dictionary
        for i, file in enumerate(spline_model_files):
            spline_model = pd.read_csv(file, index_col=0)
            if i==0:
               data_dict[sex]['Age in Days'] = spline_model['Age in Days']
            # Get region names
            region_name = file.split('cortthick-')[1]
            region_name = region_name.split(f'_{sex}.csv')[0]
            data_dict[sex][region_name] = spline_model[f'cortthick_{sex}']
    return data_dict


# Calculate statistical significance of interaction between age and gender
def calc_interaction(datapoints_all, regions):
    interaction_pvals = {}
    for reg in regions:
        # Fit a linear model where cortical thickness in this region is regressed on age and gender, including the
        # interaction between age and gender.
        model = ols(f'{reg} ~ Age_in_Days * gender', data=datapoints_all).fit()

        # Perform ANOVA
        anova_table = sm.stats.anova_lm(model, typ=2)

        interaction_pvals[reg] = anova_table.loc['Age_in_Days:gender', 'PR(>F)']

        # Print the ANOVA table
        # print(anova_table)

    # correct pvalues for multiple comparisons
    interaction_pvals_df = pd.DataFrame.from_dict(interaction_pvals, orient='index')
    uncorrected_int_ps = interaction_pvals_df[0].tolist()
    reject, corrected_int_pvals, a1, a2 = smt.multipletests(np.array(uncorrected_int_ps), alpha=0.05, method='fdr_bh')
    corrected_int_pvals_df = pd.DataFrame(corrected_int_pvals, index=regions)

    return interaction_pvals_df, corrected_int_pvals_df

def plot_data(regions, spline_models_all, datapoints_all, interaction_pvals_df, corrected_int_pvals_df, data_dir, show_plots):
    # Plot age vs cortthick for each gender for each region, both datapoints and blr model
    for reg in regions:
        fig = plt.figure()
        colors = {0: 'blue', 1: 'green'}
        sns.lineplot(data=spline_models_all, x='Age_in_Days', y=reg, hue='gender', palette=colors, legend=False)
        sns.scatterplot(data=datapoints_all, x='Age_in_Days', y=reg, hue='gender', palette=colors)
        plt.legend(title='')
        ax = plt.gca()
        fig.subplots_adjust(right=0.82)
        handles, labels = ax.get_legend_handles_labels()
        label = ['male', 'female']
        ax.legend(handles, label, loc='upper left', bbox_to_anchor=(1, 1))
        plt.title(
            f'Training Data cortthick vs. Age {reg}\n gender interaction uncorrp = '
            f'{interaction_pvals_df.loc[reg, 0]:.2f} corrp = {corrected_int_pvals_df.loc[reg, 0]:.2f}')
        plt.xlabel('Age')
        plt.ylabel('Training Data cortthick')
        if show_plots:
            plt.show(block=False)
        # Save plots to file
        plt.savefig('{}/cortthick_vs_age_withsplinefit_{}_{}_{}'
                        .format(data_dir, 'cortthick', reg, 'Training Data'))
        if show_plots == 0:
            plt.close(fig)
