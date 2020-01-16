import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Считываем данные из CSV файла с помощью функции read_csv
# В wine_train храниться обучающая выборка
# В wine_test храниться тестовая выборка
wine_train = pd.read_csv("data/wine_X_train.csv")
wine_test = pd.read_csv("data/wine_X_test.csv")

# Удаляем столбец 'quality' в xTrain,
# yTrain присваиваем значение данного стобца и
# xTest даем весь датафрэйм тестовой выборки
xTrain = wine_train.drop('quality', axis=1)
yTrain = wine_train['quality']
xTest = wine_test

# from sklearn import tree

# Была попытка в тесте на дереве решений (результаты хуже)
# clfTre = tree.DecisionTreeClassifier(max_depth=7)
# clfTre.fit(xTrain, yTrain)
# clfTre = clfTre.predict(xTest)

# print("DT (ID3) classifier results: \n", clfTre)

# print("-------------------------")

# Используем подход случайного леса (значение кол-ва деревьев выбрано )
rfc = RandomForestClassifier(n_estimators=1000, max_features=0.5)
rfc.fit(xTrain, yTrain)
pred_rfc = rfc.predict(xTest)
print("Random forest classifier results: \n", pred_rfc)

pred_rfc = np.where(pred_rfc > 6, pred_rfc, 0)
pred_rfc = np.where(pred_rfc == 0, pred_rfc, 1)
print(pred_rfc)


np.savetxt("data/wine_X_result_2.txt", pred_rfc, newline="\n", delimiter=",", fmt='%d')
