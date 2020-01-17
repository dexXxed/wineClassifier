import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split



from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import xgboost
from sklearn.metrics import accuracy_score


wine_train = pd.read_csv("data/wine_X_train.csv")
wine_test = pd.read_csv("data/wine_X_test.csv")

wine_train.quality = wine_train.quality.map({1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 1, 8: 1, 9: 1})
# print(wine_train)
# df_train_features = df.drop(['quality', 'grade'], axis=1)

# x_train, x_test, y_train, y_test = train_test_split(df_train_features, df['grade'], test_size=0.1, random_state=7)
xTrain = wine_train.drop('quality', axis=1)
yTrain = wine_train['quality']
xTest = wine_test


print('Start Predicting...')


rf = xgboost.XGBClassifier()
rf.fit(xTrain, yTrain)
rf_pred = rf.predict(xTest)


print('...Complete')

#for i in [10, 20, 30, 40, 50]: 86.25
#    rf_tune = DecisionTreeClassifier(max_depth=1)
#    rf_tune.fit(xTrain, yTrain)
#    y_pred = rf_tune.predict(xTrain)
#    print(accuracy_score(rf_pred, y_pred[320:640])*100, '%')

for i in [10, 20, 30, 40, 50]:
    rf_tune = xgboost.XGBClassifier(max_depth=int(10/4))
    rf_tune.fit(xTrain, yTrain)
    y_pred = rf_tune.predict(xTrain)
    print(accuracy_score(rf_pred, y_pred[320:640])*100, '%')

np.savetxt("data/wine_X_result_6.txt", rf_pred, newline="\n", delimiter=",", fmt='%d')
