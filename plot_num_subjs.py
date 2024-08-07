##########
# Function that plots number of subjects per age and gender
##########

import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
def plot_num_subjs(df, gender, title, struct_var, timept, path):
    sns.set(font_scale=1)
    sns.set_style(style='white')
    if gender == 'female':
        c = 'green'
    elif gender == 'male':
        c = 'blue'
    g = sns.catplot(x="age", color=c, data=df, kind="count", legend=False)
    g.fig.suptitle(title, fontsize=16)
    g.fig.subplots_adjust(top=0.85) # adjust the Figure
    g.ax.set_xlabel("Age", fontsize=12)
    g.ax.set_ylabel("Number of Subjects", fontsize=12)
    g.ax.tick_params(axis='x', labelsize=12)
    g.ax.tick_params(axis='y', labelsize=12)
    g.ax.set(yticks=np.arange(0,20,2))
    plt.show(block=False)
    plt.savefig('{}/data/{}/plots/NumSubjects_{}_{}'.format(path, struct_var, struct_var, timept))
