from ml_uav_project.EDA.preprocess import plot_model_results, show_model_results
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error


def linear_regression(X_train, y_train, X_test, y_test):
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    r_squared_error = lin_reg.score(X_train, y_train)
    print("R-Squared error on training dataset is: " , r_squared_error)
    #predicted_train = lin_reg.predict(X_train_scaled)
    #train_error = mean_squared_error(y_train,predicted_train,squared = False)
    predicted_test = lin_reg.predict(X_test)
    test_error = mean_squared_error(y_test, predicted_test, squared = False)
    print("Test RMSE error is: ", test_error, " dB.")
    r_squared_error_test = lin_reg.score(X_test, y_test)
    print("R-Squared error on test dataset is:",r_squared_error_test)
    test_MAPE = mean_absolute_percentage_error(y_test, predicted_test)
    print("Test MAPE error is:", test_MAPE)
    test_MAE = mean_absolute_error(y_test, predicted_test)
    print("Test MAE error is:", test_MAE)
    results = (test_error, r_squared_error_test, test_MAPE, test_MAE)
    plot_model_results(y_test, predicted_test, "Linear Regression")
    return results


