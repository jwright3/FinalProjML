import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.preprocessing as poly
import statsmodels.api as sm
import seaborn as sns
from sklearn.model_selection import cross_val_score, LeaveOneOut, train_test_split
from sklearn import model_selection, linear_model, feature_selection, metrics
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


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
    Y_train = Y_train.values.ravel()
    n_est_scores = []
    n_est_values = []
    for n_est in range(10, 400, 10):
        random_forest_classifier = RandomForestClassifier(n_estimators=n_est)

        random_forest_classifier.fit(X_train, Y_train)
        y_predict = random_forest_classifier.predict(X_test)
        score = sk.metrics.accuracy_score(Y_test, y_predict)
        n_est_scores.append(score)
        n_est_values.append(n_est)
        print(f"n_est: {n_est} Score: {score}")

    plt.plot(n_est_values, n_est_scores)
    plt.title("Num Estimators Tuning")
    plt.xlabel("num_estimators")
    plt.ylabel("Score")
    plt.grid()
    plt.show()

def knn(X_train, Y_train, X_test, Y_test):
    #making models
    knn_model = KNeighborsClassifier(n_neighbors=1).fit(X_train, Y_train)
    knn_5 = KNeighborsClassifier(n_neighbors=5).fit(X_train, Y_train)
    knn_10 = KNeighborsClassifier(n_neighbors=10).fit(X_train, Y_train)

    #getting predictors
    knn_preds = knn_model.predict(X_test)
    knn_preds_5 = knn_5.predict(X_test)
    knn_preds_10 = knn_10.predict(X_test)

    #getting accuracy scores and printing
    accuracy_score = sk.metrics.accuracy_score(Y_test, knn_preds)
    accuracy_score5 = sk.metrics.accuracy_score(Y_test, knn_preds_5)
    accuracy_score10 = sk.metrics.accuracy_score(Y_test, knn_preds_10)
    print(accuracy_score)
    print(accuracy_score5)
    print(accuracy_score10)

if __name__ == '__main__':
    # Data cleaning and feature creation

    health_data = pd.read_csv('./health_data.csv')
    health_data.dropna(inplace=True)

    # print(health_data.columns)

    #pairplot generation and display
    sns.pairplot(health_data)
    plt.show()

    x = health_data.loc[:, ~health_data.columns.isin(['Hypertension', 'Diabetes', 'Stroke'])]
    y = health_data.loc[:, ['Diabetes']]

    # print(x)
    # print(health_data)

    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2)

    
    # Algorithms

    # Logistic Regression
    logRegression(X_train, Y_train, X_test, Y_test)

    # Scores of our logistic regression cross validated with n = 5 k-folds
    scores = cross_val_score(logRegression, x, y, scoring="accuracy", cv=5)

    print(scores)

    # KNN
    knn(X_train, Y_train, X_test, Y_test)
    scores = cross_val_score(knn, x, y, scoring="accuracy", cv=5)

    print(scores)
    # RandomForest
    randomForestRegression(X_train, Y_train, X_test, Y_test)
