import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


def load_data(file_loc="ml_uav_project/ml_uav_project/ml_uav_project_dataset.xlsx"):
    df = pd.read_excel(file_loc)
    return df


pathloss_df = load_data()
print(pathloss_df.head(5))
