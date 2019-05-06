#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""======================================================================"""
"""IDE used: Spyder
   Python 3.6.4 on Anaconda3"""
#============================================================================
"""
Created on Sat Feb 24 15:28:30 2018

Due Date: Thursday March 01 23:59 2018

@author: Alem Haddush Fitwi
Email:afitwi1@binghamton.edu
Neural Network & Deep Learning - EECE680C
Department of Electrical & Computer Engineering
Watson Graduate School of Engineering & Applied Science
The State University of New York @ Binghamton
"""
#============================================================================
"""
CNN-MNIST (0 ~ 9) Classifier
Goal: To classify Image dataset into 10 classes of digits (0 ~ 10)
"""
#============================================================================
#----------------------------------------------------------------------------
"""
Step_1: Importing Required Packages or modules:
"""
#----------------------------------------------------------------------------
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import idx2numpy
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import regularizers
from keras import metrics
from keras.utils.np_utils import to_categorical
from keras import optimizers
from scipy import misc
#============================================================================
#----------------------------------------------------------------------------
"""
Step_2: Data Acquisition and preproccessing:
"""
#----------------------------------------------------------------------------
# Dataset is downloaded from http://yann.lecun.com/exdb/mnist/
DATA_PATH = 'data/'
# conver idx to np format and load the four datasets
X_train = idx2numpy.convert_from_file(DATA_PATH + 'train-images.idx3-ubyte')
Y_train = idx2numpy.convert_from_file(DATA_PATH + 'train-labels.idx1-ubyte')
X_test = idx2numpy.convert_from_file(DATA_PATH + 't10k-images.idx3-ubyte')
Y_test = idx2numpy.convert_from_file(DATA_PATH + 't10k-labels.idx1-ubyte')
#Reshape the data so as to fit the format of (samples, height, width, channels)
X_train = X_train.reshape(60000, 28, 28, 1).astype('float32')
X_test = X_test.reshape(10000, 28, 28, 1).astype('float32')
Y_train = Y_train.reshape(60000)
Y_test = Y_test.reshape(10000)
Y_train = to_categorical(Y_train, 10)
Y_test = to_categorical(Y_test, 10)
#============================================================================
#----------------------------------------------------------------------------
"""
Step_3: Model Building and Definition:
"""
#----------------------------------------------------------------------------
model = Sequential()
model.add(Conv2D(filters=20, kernel_size=(6,6), 
                 kernel_regularizer=regularizers.l2(0.04), strides=(1,1), 
                 padding='valid', activation='relu', 
                 data_format='channels_last', input_shape=(28,28,1)))
model.add(Conv2D(filters=20, kernel_size=(3,3), 
                 kernel_regularizer=regularizers.l2(0.04), strides=(1,1), 
                 padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(4,4), strides=(1,1)))
model.add(Dropout(rate=0.05,seed=3))
model.add(Conv2D(filters=10, kernel_size=(6,6), 
                 kernel_regularizer=regularizers.l2(0.04), strides=(1,1), 
                 padding='valid', activation='relu'))
model.add(Conv2D(filters=10, kernel_size=(3,3), 
                 kernel_regularizer=regularizers.l2(0.04), strides=(1,1), 
                 padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(4,4), strides=(1,1)))
model.add(Dropout(rate=0.05,seed=8))
model.add(Flatten())
model.add(Dense(units=30, activation='tanh', 
                kernel_regularizer=regularizers.l2(0.04)))
model.add(Dense(units=10, activation='softmax', 
                kernel_regularizer=regularizers.l2(0.04)))
#============================================================================
#----------------------------------------------------------------------------
"""
Step_4: Model Compilation:
"""
#----------------------------------------------------------------------------
#Reduce the learning rate if training accuracy suddenly drops and keeps 
#  decreasing
sgd = optimizers.SGD(lr=0.003) # lr by default is 0.01 for SGD
model.compile(loss='categorical_crossentropy', optimizer=sgd, 
              metrics=[metrics.categorical_accuracy])

#============================================================================
#----------------------------------------------------------------------------
"""
Step_5: Model Fitting:
"""
#----------------------------------------------------------------------------
model.fit(X_train, Y_train, epochs=5, batch_size=50)
model.save('mnist-classifier-model.h5')
model.save_weights('mnist-classifier-weights.h5')
#============================================================================
#----------------------------------------------------------------------------
"""
Step_6: Model Evaluation:
"""
#----------------------------------------------------------------------------
print("\nEvaluating the model on test data. This won't take long. Relax!")
test_loss, test_accuracy = model.evaluate(X_test, Y_test, batch_size=10)
print("\nAccuracy on test data : ", test_accuracy*100)
print("\nLoss on test data : ", test_loss)
#============================================================================
#----------------------------------------------------------------------------
"""
Step_7: Ending the Session:
"""
#----------------------------------------------------------------------------
#Manually end the session to avoid occasional exceptions while running the 
#  ... program
from keras import backend as K
K.clear_session()
#============================================================================
#----------------------------------------------------------------------------
"""                            End of Program                             """
#----------------------------------------------------------------------------
"""  Datasets were downloaded from http://yann.lecun.com/exdb/mnist/  """