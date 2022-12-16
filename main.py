import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as poly
import statsmodels.api as sm
import seaborn as sns
from sklearn.model_selection import cross_val_score, LeaveOneOut, train_test_split
from sklearn import model_selection, linear_model, feature_selection, metrics
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


def genConfusionMatrix(Y_test, Y_predict):
    print("\t\t Confusion Matrix Log Regression Diabetes")
    print("\t Below Above(Predicted)")
    print("Below \t" + str(confusion_matrix(y_true=Y_test, y_pred=Y_predict)[0]))
    print("Above \t" + str(confusion_matrix(y_true=Y_test, y_pred=Y_predict)[1]))

    print("Accuracy of logistic regression")


def logRegression(X_train, Y_train, X_test, Y_test):
    logRegression = LogisticRegression()
    Y_predict = logRegression.fit(X_train, Y_train).predict(X_test)

    genConfusionMatrix(Y_test, Y_predict)
    # Scores of our logistic regression cross validated with n = 5 k-folds
    scores = cross_val_score(logRegression, x, y, scoring="accuracy", cv=5)

    print(scores)


def randomForestRegression(X_train, Y_train, X_test, Y_test):
    random_forest_classifier = RandomForestClassifier(n_estimators=19)

    random_forest_classifier.fit(X_train, Y_train)
    y_predict = random_forest_classifier.fit(X_test)


if __name__ == '__main__':
    # Data cleaning and feature creation

    health_data = pd.read_csv('./health_data.csv')
    health_data.dropna(inplace=True)

    print(health_data.columns)

    x = health_data.loc[:, ~health_data.columns.isin(['Hypertension', 'Diabetes', 'Stroke'])]
    y = health_data.loc[:, ['Diabetes']]

    print(x)
    print(y)
    print(health_data)

    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2)

    # Algorithms

    # Logistic Regression
    logRegression(X_train, Y_train, X_test, Y_test)

    # Scores of our logistic regression cross validated with n = 5 k-folds
    scores = cross_val_score(logRegression, x, y, scoring="accuracy", cv=5)

    print(scores)

    # KNN

    # RandomForest