# -*- coding: utf-8 -*-
"""Hyperparameters of ANN_Chrun_KN

"""
import keras as ks
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# import the data...
churn_d = pd.read_csv('file:///C:/Users/admin/Desktop/Deep_learning/ANN/Churn_Modelling.csv')

churn_d.describe
churn_d.columns
churn_d.shape

# check null values...
churn_d.isnull().sum()   # '0' Null values

# Features & Labels...
X = churn_d.iloc[:, 3:13]    # remove unwanted columns...
y = churn_d.iloc[:, 13]      

X.describe
y.describe
X.groupby('Geography').first()

# Create dummy variables to convert text string format into integer data...
geography = pd.get_dummies(X['Geography'],drop_first=True)
gender = pd.get_dummies(X['Gender'],drop_first=True)

geography.head

# concatenate the dataframes
X = pd.concat([X,geography,gender],axis=1)

X.describe
X.columns

X.drop(['Geography','Gender'],axis=1,inplace=True)   #remove original columns which are converted to dummy

X.describe
X.columns

# Split the data into train & test...
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()       #scale down the input features into same scale range.
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

""" **Make ANN Model**
Perform Hyperparameter Optimization...
"""

from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten, BatchNormalization
from keras.layers import PReLU, ELU
from keras.activations import relu, sigmoid
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def create_model(layers, activation):
  model = Sequential()
  for i, nodes in enumerate(layers):    
    if i==0:
      model.add(Dense(nodes,input_dim=X_train.shape[1]))      # for input layer
      model.add(Dropout(0.3))
    else:
      model.add(Dense(nodes))                                # for hidden layer
      model.add(Activation(activation))
      model.add(Dropout(0.3))

  model.add(Dense(units=1, kernel_initializer='glorot_uniform',activation='sigmoid'))  # for o/p layer
  
  model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
  return model

model = KerasClassifier(build_fn=create_model, verbose=0)

# Parameters...
layers = [[20],[40,20],[45,30,15]]   # from these 3 choose best one.
activations = ['sigmoid','relu']     # best one activation function
param_grid = dict(layers=layers, activation=activations, batch_size=[128,256],epochs=[30])
grid = GridSearchCV(estimator=model, param_grid=param_grid,cv=5)
# cv=5 means for 5 no. of iterations.

# Fitting the ANN into Training set...
grid_result = grid.fit(X_train,y_train)

# Model test results
print(grid_result.best_score_,grid_result.best_params_)
#Results...0.8587142868041993 
#{'activation': 'relu', 'batch_size': 128, 'epochs': 30, 'layers': [40, 20]}


### Making the predictions and evaluating the model
# Predicting the Test set results
pred_y = grid.predict(X_test)
y_pred = (pred_y > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm    
#array([[2291,   91],
#       [ 343,  275]]

# Calculate the Accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred,y_test)
score      # 85.33

### so we got 85.87% from hyperparameter results & 85.33% test accuracy...
# so we choose these parameters for better results.