import ml_uav_project
from ml_uav_project.EDA import load_data, export_features_statistics, plot_feature_kde
from ml_uav_project.LinearRegression import linear_regression
from sklearn.model_selection import train_test_split,cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler


# Load dataset as pathloss_df DataFrame
pathloss_df = load_data()

# Print features list and KDE Plots
features_list = pathloss_df.columns
for i in range(13):
    col = features_list[i]
    #plot_feature_kde(pathloss_df,col) Uncomment to plot KDE_plot of features as png
    print(col)

# Export Features statistics
#export_features_statistics(pathloss_df, file_loc="ml_uav_project/Data/features_statistics.xlsx")

# Test Train Split
X = pathloss_df.drop(["Path_loss_average"],axis=1).values
y = pathloss_df["Path_loss_average"].values
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=42, shuffle = True)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

## 1. Linear Regression
linear_regression(X_train=X_train_scaled, y_train=y_train)


