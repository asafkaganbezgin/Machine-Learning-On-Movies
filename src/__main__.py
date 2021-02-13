import polyreg

import os
import operator
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import polyreg

if __name__ == "__main__":

    rootPath = os.path.abspath(os.path.dirname(__file__))
    dataPath = os.path.join("dataset/processed/train.csv")

    data = pd.read_csv(dataPath)

    # Plotting an histogram on revenue which is the target(success of a movie).
    sns.set(rc={"figure.figsize": (11.7, 8.27)})
    sns.distplot(data["revenue"], bins=30)
    # plt.show()

    # Dropping the some columns becase they are not related with project
    data = data.drop(["id"], axis=1)
    data = data.drop(["belongs_to_collection"], axis=1)
    data = data.drop(["genres"], axis=1)
    data = data.drop(["homepage"], axis=1)
    data = data.drop(["imdb_id"], axis=1)
    data = data.drop(["original_language"], axis=1)
    data = data.drop(["original_title"], axis=1)
    data = data.drop(["overview"], axis=1)
    data = data.drop(["poster_path"], axis=1)
    data = data.drop(["production_companies"], axis=1)
    data = data.drop(["production_countries"], axis=1)
    data = data.drop(["release_date"], axis=1)
    data = data.drop(["spoken_languages"], axis=1)
    data = data.drop(["status"], axis=1)
    data = data.drop(["tagline"], axis=1)
    data = data.drop(["title"], axis=1)
    data = data.drop(["Keywords"], axis=1)
    data = data.drop(["cast"], axis=1)
    data = data.drop(["crew"], axis=1)

    # Plotting an heatmap which displays the correlation of features with the target.
    correlation_matrix = data.corr().round(2)
    sns.heatmap(data=correlation_matrix, annot=True)

    # Plotting the datapoints belong to the feaures with respect to the target.
    plt.figure(figsize=(20, 5))

    features = ["budget", "popularity", "runtime"]
    target = data["revenue"]

    for i, col in enumerate(features):
        plt.subplot(1, len(features), i + 1)
        x = data[col]
        y = target
        plt.scatter(x, y, marker="o")
        plt.title(col)
        plt.xlabel(col)
        plt.ylabel("revenue")

    # plt.show()

    # Checking if there is a null value in a feature column
    print(data.isnull().sum())
    print("")

    # Preparing the data for training. 90:10 split ratio is applied
    X = pd.DataFrame(np.c_[data['budget'], data['popularity'], data['runtime']], columns=[
                     'budget', 'popularity', 'runtime'])
    Y = pd.DataFrame(np.c_[data['revenue']], columns=['revenue'])

    scaler = MinMaxScaler(feature_range=(0.1,1))
    rescaledX = scaler.fit_transform(X)

    scaler = MinMaxScaler(feature_range=(0.1,1))
    rescaledY = scaler.fit_transform(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(
        rescaledX, rescaledY, test_size=0.1, random_state=5)

    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_test.shape)
    print("")

    # Applying polynomial regression on the data. The first parameter is the
    # degree of the regression. 2 means there will be X^2 terms
    polyreg.modelPolynomialRegression(3, X_train, X_test, Y_train, Y_test)
