# -*- coding: utf-8 -*-
"""ANN_Churn_KN
"""

import keras as ks
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Import the data...
churn_d = pd.read_csv('/content/drive/My Drive/#Data Science/Deep Learning/ANN/Churn_Modelling.csv')

churn_d.describe
churn_d.columns
churn_d.shape

# check null values...
churn_d.isnull().sum()   # '0' Null values

#Features and Labels...
X = churn_d.iloc[:, 3:13]
y = churn_d.iloc[:, 13]

X.describe
y.describe
X.groupby('Geography').first()

# Create dummy variables to convert  text categorial data into numerical format. 
geography = pd.get_dummies(X['Geography'],drop_first=True)
gender = pd.get_dummies(X['Gender'],drop_first=True)

geography.head

# concatenate the dataframes
X = pd.concat([X,geography,gender],axis=1)

X.describe
X.columns

#drop original columns which we get as dummy...
X.drop(['Geography','Gender'],axis=1,inplace=True)

X.describe
X.columns

# split the data into train & test...
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


""" Make ANN Model """

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU, PReLU, ELU
from keras.layers import Dropout

# Initialize the Sequential library...
classifier = Sequential()   #so it is an empty Neural Network currently

# Add input layer & first hidden layer...
classifier.add(Dense(units=6, kernel_initializer='he_uniform',activation='relu',input_dim=11))

# Add 2nd hidden layer...
classifier.add(Dense(units=6, kernel_initializer='he_uniform',activation='relu'))

# Adding the output layer...
classifier.add(Dense(units=1,kernel_initializer='glorot_uniform',activation='sigmoid'))

classifier.summary()

# Compiling the ANN...
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# Fitting the ANN into Training set...
model_history = classifier.fit(X_train,y_train,validation_split=0.30,batch_size=10,nb_epoch=100)

# list all data in history
print(model_history.history.keys())

# summarize history for Accuracy
plt.plot(model_history.history['acc'])
plt.plot(model_history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

# Calculate the Accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred,y_test)
score     # 85.57%