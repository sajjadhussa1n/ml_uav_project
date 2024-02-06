import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(file_loc="ml_uav_project/ml_uav_project/ml_uav_project_dataset.xlsx"):
    df = pd.read_excel(file_loc)
    return df


def export_features_statistics(df, file_loc="ml_uav_project/ml_uav_project/features_statistics.xlsx"):
    new_df = df.describe().transpose()
    new_df.to_excel(file_loc)

def plot_feature_kde(df, col="Phi", xlab="Phi"):
    sns.kdeplot(data=df, x=col, fill=True, palette="crest", alpha=.75, linewidth=0)
    #ax1.set_xlabel(xlab)
    #ax1.set_ylabel("Count")
    plt.savefig("ml_uav_project/ml_uav_project/EDA/"+col+"_kde_plot.eps",dpi=600)
    
    


pathloss_df = load_data()
# export_features_statistics(pathloss_df)
plot_feature_kde(pathloss_df)
