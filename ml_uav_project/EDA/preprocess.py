import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_loc="ml_uav_project/Data/ml_uav_project_dataset.xlsx"):
    df = pd.read_excel(file_loc)
    return df


def export_features_statistics(df, file_loc="ml_uav_project/Data/features_statistics.xlsx"):
    new_df = df.describe().transpose()
    new_df.to_excel(file_loc)


def plot_feature_kde(df, col="Phi"):
    plt.figure(figsize=(10, 8), layout = 'tight')
    sns.set_palette('colorblind')
    sns.kdeplot(data=df, x=col, cut=0, color='cadetblue', fill=True, alpha=0.5)
    plt.xlabel(col, fontsize=22)
    plt.ylabel("Density", fontsize=22)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.savefig("ml_uav_project/Data/"+col+"_kde_plot.png", dpi=600) 


pathloss_df = load_data()
features_list = pathloss_df.columns
for i in range(13):
    print(features_list[i])


# export_features_statistics(pathloss_df)
#plot_feature_kde(pathloss_df)
