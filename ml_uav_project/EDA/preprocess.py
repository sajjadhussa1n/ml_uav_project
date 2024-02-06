import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


def load_data(file_loc="ml_uav_project/ml_uav_project/ml_uav_project_dataset.xlsx"):
    df = pd.read_excel(file_loc)
    return df


def export_features_statistics(df, file_loc="ml_uav_project/ml_uav_project/features_statistics.xlsx"):
    new_df = df.describe().transpose()
    new_df.to_excel(file_loc)


pathloss_df = load_data()
export_features_statistics(pathloss_df)
