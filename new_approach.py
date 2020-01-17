import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
# from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

wine = pd.read_csv("data/wine_X_train.csv")
wine_test = pd.read_csv("data/wine_X_test.csv")


bins = (2, 6.5, 10)
group_names = ['bad', 'good']
wine['quality'] = pd.cut(wine['quality'], bins=bins, labels=group_names)  # cutting it by the bins, using labels
wine['quality'].unique()
label_quality = LabelEncoder()
wine['quality'] = label_quality.fit_transform(wine['quality'])
print(wine)
print(wine['quality'].value_counts())

# Now separate the dataset as response variable and feature variables
X = wine.drop('quality', axis=1)
y = np.where(wine['quality'] > 6, 1, 0)
xTest = wine_test

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)


rfc = RandomForestClassifier(n_estimators=100, max_features=10)
rfc.fit(X, y)
pred_rfc = rfc.predict(xTest)

# accuracy = rfc.score(X_test, y_test)
# print(f'ok = {accuracy}')
# from sklearn.metrics import accuracy_score
# cm = accuracy_score(y_test, pred_rfc)
print(pred_rfc)

np.savetxt("data/wine_X_result_new.txt", pred_rfc, newline="\n", delimiter=",", fmt='%d')
