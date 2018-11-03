# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 22:06:26 2018

@author: Mohammed Ali
"""
# importing the libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reading data from the data set
dataset_train = pd.read_csv('data_train.csv')
dataset_test = pd.read_csv('data_test.csv')


# Filling the missing values for the non-categorical variables
dataset_train.iloc[:,1:43] = dataset_train.iloc[:,1:43].interpolate(method = 'linear')
dataset_test.iloc[:,1:43] = dataset_test.iloc[:,1:43].interpolate(method = 'linear')

#Filling the missing values for the categorical variables
from fancyimpute import KNN
dataset_train.iloc[:,43:57] = KNN(k=3).fill(dataset_train.iloc[:,43:57], np.isnan(dataset_train.iloc[:,43:57]))
dataset_test.iloc[:,43:57] = KNN(k=3).fill(dataset_test.iloc[:,43:57], np.isnan(dataset_test.iloc[:,43:57]))


# creating X and y variables for training set
X_train = dataset_train.iloc[:, 0:57]
y_train = dataset_train.iloc[:,57]

#Creating X_test

X_test = dataset_test.iloc[:, 0:57]


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Fitting the decision tree classification model to the training data set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'gini', random_state = 0)
classifier.fit(X_train, y_train)

from sklearn.metrics import f1_score
F1 = f1_score(y_test, y_pred, average='macro')
print('F1 score is: ',F1)


# Creating X and y variables for test set for validation

X_test = dataset_test.iloc[:,0:57]

# Fitting a model to the dataset




