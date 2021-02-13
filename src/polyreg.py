import numpy as np

import sklearn.metrics as metric
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def modelPolynomialRegression(degree, X_train, X_test, Y_train, Y_test):

    polyFeatures = PolynomialFeatures(degree=degree)

    # Transforming the features to higher order features.
    x_train_poly = polyFeatures.fit_transform(X_train)

    # Fitting the transformed features into Linear Regression
    polyModel = LinearRegression()
    polyModel.fit(x_train_poly, Y_train)

    # Making a prediction on the training data set.
    y_train_predicted = polyModel.predict(x_train_poly)

    # Making a prediction on the test data set.
    y_test_predicted = polyModel.predict(polyFeatures.fit_transform(X_test))

    # Making an evaluation about the model on training dataset.
    mseTrain = metric.mean_squared_error(Y_train, y_train_predicted)
    rmseTrain = np.sqrt(metric.mean_squared_error(Y_train, y_train_predicted))
    msleTrain = metric.mean_squared_log_error(Y_train, y_train_predicted)
    rmsleTrain = np.sqrt(metric.mean_squared_log_error(Y_train, y_train_predicted))
    r2_train = metric.r2_score(Y_train, y_train_predicted)

    # Making an evaluation about the model on test dataset
    mseTest = metric.mean_squared_error(Y_test, y_test_predicted)
    rmseTest = np.sqrt(metric.mean_squared_error(Y_test, y_test_predicted))
    msleTest = metric.mean_squared_log_error(Y_test, y_test_predicted)
    rmsleTest = np.sqrt(metric.mean_squared_log_error(Y_test, y_test_predicted))
    r2_test = metric.r2_score(Y_test, y_test_predicted)

    print("Performance metrics of training dataset:")
    print("**********************************")
    print("MSE of the training dataset is {}".format(mseTrain))
    print("RMSE of the training dataset is {}".format(rmseTrain))
    print("MSLE of the training dataset is {}".format(msleTrain))
    print("RMSLE of the training dataset is {}".format(rmsleTrain))
    print("R2 score of training set is {}".format(r2_train))
    print("")
    print("Performance metrics of the test dataset:")
    print("**********************************")
    print("MSE of the test set is {}".format(mseTest))
    print("RMSE of the test set is {}".format(rmseTest))
    print("MSLE of the test set is {}".format(msleTest))
    print("RMSLE of the test set is {}".format(rmsleTest))
    print("R2 score of test set is {}".format(r2_test))
    print("")