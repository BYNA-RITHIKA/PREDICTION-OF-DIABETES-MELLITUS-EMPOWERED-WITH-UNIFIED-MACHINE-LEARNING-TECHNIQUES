import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import pickle

dataset = pd.read_csv('diabetes.csv')
dataset_new = dataset
dataset_new[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] = dataset_new[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",]].replace(0, np.NaN) 
dataset_new["Glucose"].fillna(dataset_new["Glucose"].mean(), inplace = True)
dataset_new["BloodPressure"].fillna(dataset_new["BloodPressure"].mean(), inplace = True)
dataset_new["SkinThickness"].fillna(dataset_new["SkinThickness"].mean(), inplace = True)
dataset_new["Insulin"].fillna(dataset_new["Insulin"].mean(), inplace = True)
dataset_new["BMI"].fillna(dataset_new["BMI"].mean(), inplace = True)

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
dataset_scaled = sc.fit_transform(dataset_new)
dataset_scaled = pd.DataFrame(dataset_scaled)
X = dataset_scaled.iloc[:, [1, 4, 5, 7]].values
Y = dataset_scaled.iloc[:, 8].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 42, stratify = dataset_new['Outcome'] )

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(random_state = 42)
logreg.fit(X_train, Y_train)
y_pred=logreg.predict(X_test)

from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors = 24, metric = 'minkowski', p = 2)
knn_model.fit(X_train, Y_train)
y_pred_knn=knn_model.predict(X_train)

from sklearn.svm import SVC
svc = SVC(kernel = 'linear', random_state = 42)
svc.fit(X_train, Y_train)
y_pred_svc=svc.predict(X_test)

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, Y_train)
y_pred_nb=nb.predict(X_test)

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier(criterion = 'entropy',max_depth=6, random_state = 42)
decision_tree.fit(X_train, Y_train)
y_pred_dt=decision_tree.predict(X_test)

from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators = 23, criterion = 'entropy',max_depth=6, random_state = 42)
random_forest.fit(X_train, Y_train)
y_pred_rf=random_forest.predict(X_test)

pickle.dump(logreg, open('logreg_model.pkl','wb'))
pickle.dump(knn_model, open('knn_model.pkl','wb'))
pickle.dump(svc, open('svc_model.pkl','wb'))
pickle.dump(nb, open('nb_model.pkl','wb'))
pickle.dump(decision_tree, open('decision_tree.pkl','wb'))
pickle.dump(random_forest, open('random_forest.pkl','wb'))