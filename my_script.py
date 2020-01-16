from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np


# Считываем данные из CSV файла с помощью функции read_csv
# В wine_train храниться обучающая выборка
# В wine_test храниться тестовая выборка
wine_train = pd.read_csv("data/wine_X_train.csv")
wine_test = pd.read_csv("data/wine_X_test.csv")

xTrain = wine_train[list(wine_train)[:-1]]
yTrain = wine_train['quality']
xTest = wine_test

regressor = LinearRegression()

regressor.fit(xTrain, yTrain)
ypred = regressor.predict(xTest)
print(ypred)
print(len(ypred))

ypred = np.where(ypred > 6, ypred, 0)
print(ypred)
ypred = np.where(ypred == 0, ypred, 1)
print(ypred)

np.savetxt("data/wine_X_result_4.txt", ypred, newline="\n", delimiter=",", fmt='%d')
