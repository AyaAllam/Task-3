# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 23:44:01 2021

@author: HP
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
dataset = pd.read_csv('diabetes.csv')
x=dataset.iloc[:,0:8].values
y=dataset.iloc[:,8].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)
#Decision Tree
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier()
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
ac=(cm[0][0]+cm[1][1])/len(y_test)
print('The decision tree model accuracy =',ac*100,'%')
#Random Forest
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier()
classifier.fit(x_train,y_train)   
y_pred=classifier.predict(x_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
ac=(cm[0][0]+cm[1][1])/len(y_test)
print('The random forest model accuracy=',ac*100,'%')
#Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
ac=(cm[0][0]+cm[1][1])/len(y_test)
print("The logistic regression model accuracy=",ac*100,"%")
#KNN
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 8)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
ac=(cm[0][0]+cm[1][1])/len(y_test)
print("The logistic regression model accuracy=",ac*100,"%")




