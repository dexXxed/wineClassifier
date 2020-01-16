import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
import graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

wine_train = pd.read_csv("data/wine_X_train.csv")
wine_test = pd.read_csv("data/wine_X_test.csv")

xTrain = wine_train.drop('quality', axis=1)
yTrain = wine_train['quality']
xTest = wine_test

print(xTrain)
print(yTrain)
print(xTest)

clfTre = tree.DecisionTreeClassifier(max_depth=7)
clfTre.fit(xTrain, yTrain)
clfTre = clfTre.predict(xTrain)

print(clfTre)

print ("-------------------------")

rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(xTrain, yTrain)
pred_rfc = rfc.predict(xTest)
print(pred_rfc)




