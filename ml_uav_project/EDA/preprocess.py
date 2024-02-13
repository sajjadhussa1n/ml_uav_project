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


def plot_model_results(y_test, predicted_test, model_name='Linear Regression', format='eps'):
    df = pd.DataFrame({ 'True_values': y_test, 'Predicted_values':predicted_test })
    true_value = df["True_values"].reset_index(drop=True)
    predicted_value = df["Predicted_values"].reset_index(drop=True)
    plt.figure(figsize=(10, 8))
    plt.scatter(true_value, predicted_value, c='crimson',marker = '+',s=100)
    p1 = max(max(predicted_value), max(true_value))
    p2 = min(min(predicted_value), min(true_value))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('True Values (dB)', fontsize=22)
    plt.ylabel('Predictions (dB)', fontsize=22)
    plt.title(model_name, fontsize=22)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.grid()
    short_name = model_name.lower().split()
    plt.savefig("ml_uav_project/Data/"+short_name[0]+"_"+short_name[-1]+"_results."+format, dpi=600)
    plt.show()


def show_model_results(y_test, predicted_test, model_name='Linear Regression'):
    df = pd.DataFrame({ 'True_values': y_test, 'Predicted_values':predicted_test })
    true_value = df["True_values"].reset_index(drop=True)
    predicted_value = df["Predicted_values"].reset_index(drop=True)
    plt.figure(figsize=(10, 8))
    plt.scatter(true_value, predicted_value, c='crimson',marker = '+',s=100)
    p1 = max(max(predicted_value), max(true_value))
    p2 = min(min(predicted_value), min(true_value))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('True Values (dB)', fontsize=22)
    plt.ylabel('Predictions (dB)', fontsize=22)
    plt.title(model_name, fontsize=22)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.grid()
    plt.show()