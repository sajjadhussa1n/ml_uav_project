from sklearn.linear_model import LinearRegression


def linear_regression(X_train, y_train):
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    r_squared_error = lin_reg.score(X_train, y_train)
    print("R-Squared error on training dataset is: " , r_squared_error)
    #predicted_train = lin_reg.predict(X_train_scaled)
    #train_error = mean_squared_error(y_train,predicted_train,squared = False)