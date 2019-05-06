#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""======================================================================"""
"""IDE used: Spyder
   Python 3.6.4 on Anaconda3"""
#============================================================================
"""
Created on Created on Mon Feb 19 17:44:16 2018

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
Iris Type Classifier
Task Goal: To classify dataset into three classes
Every input in the  Iris dataset comprises the following attributes: 
 1) Id 
 2) SepalLengthCm 
 3) SepalWidthCm
 4) PetalLengthCm 
 5) PetalWidthCm 
 6) Species/Class Label (Iris-Setosa, Iris-Versicolor,Iris-virginica)  
"""
#============================================================================
#----------------------------------------------------------------------------
"""
Step_1: Importing Required Packages or modules:
"""
#----------------------------------------------------------------------------
import tensorflow as tf
import pandas as pd
import numpy as np
#============================================================================
#----------------------------------------------------------------------------
"""
Step_2: Reading or loading the Iris Dataset, named Iris_Dataset.csv
        Please place python code and the dataset on the same folder
"""
#----------------------------------------------------------------------------
dataset = pd.read_csv('Iris_Dataset.csv')
dataset = pd.get_dummies(dataset, columns=['ClassLabel']) # can be Species
# One Hot Encoding for convenience
values = list(dataset.columns.values)
y = dataset[values[-3:]]
y = np.array(y, dtype='float32')
X = dataset[values[1:-3]]
X = np.array(X, dtype='float32')
#============================================================================
#----------------------------------------------------------------------------
"""
Step_4: Reshuffling the data"""
#----------------------------------------------------------------------------
indices = np.random.choice(len(X), len(X), replace=False)
X_values = X[indices]
y_values = y[indices]
#============================================================================
#----------------------------------------------------------------------------
"""
Step_5: Breaking the dataset into training and test datasets"""
#----------------------------------------------------------------------------
test_size = 10
X_test = X_values[-test_size:]
X_train = X_values[:-test_size]
y_test = y_values[-test_size:]
y_train = y_values[:-test_size]
#============================================================================
#----------------------------------------------------------------------------
"""
Step_6: Creating the tensor flow session"""
#----------------------------------------------------------------------------
sess = tf.Session()
#============================================================================
#----------------------------------------------------------------------------
"""
Step_7: Define the number of intervals and epoches"""
#----------------------------------------------------------------------------
interval = 50 # Results will be displayed at interval of 50 Epoches
epoch = 500
#============================================================================
#----------------------------------------------------------------------------
"""
Step_8: Create the ANN model or achitectureby initializing placeholders
        Define four input neurons for the four inputs:
            1) Sepal Length 2) Sepal Width
            3) Petal Length 4) Petal Width
        Define Four output neurons for the three species/Class labels:
            1) Iris-Setosa 2) Iris-Versicolor 3) Iris-virginica
        As recommended in many previous works, we can use 8 Hidden Neurons
            """
#----------------------------------------------------------------------------
X_data = tf.placeholder(shape=[None, 4], dtype=tf.float32) 
hidden_layer_nodes = 8 
y_target = tf.placeholder(shape=[None, 3], dtype=tf.float32) 
w1 = tf.Variable(tf.random_normal(shape=[4,hidden_layer_nodes])) 
b1 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]))   
w2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes,3])) 
b2 = tf.Variable(tf.random_normal(shape=[3]))   
#============================================================================
#----------------------------------------------------------------------------
"""
Step_9: Define the operations of the hidden and output layers"""
#----------------------------------------------------------------------------
hidden_output = tf.nn.relu(tf.add(tf.matmul(X_data, w1), b1))
final_output = tf.nn.softmax(tf.add(tf.matmul(hidden_output, w2), b2))
#============================================================================
#----------------------------------------------------------------------------
"""
Step_10: Define the cost function"""
#----------------------------------------------------------------------------
loss = tf.reduce_mean(-tf.reduce_sum(y_target * tf.log(final_output), axis=0))
#============================================================================
#----------------------------------------------------------------------------
"""
Step_11: Define an optimizer"""
#----------------------------------------------------------------------------
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
#============================================================================
#----------------------------------------------------------------------------
"""
Step_12: Initialization of pertinent variables"""
#----------------------------------------------------------------------------
init = tf.global_variables_initializer()
sess.run(init)
print()
#============================================================================
#----------------------------------------------------------------------------
"""
Step_13: Define the Training here"""
#----------------------------------------------------------------------------
print("-------------------****************-------------------------")
print("---------Iris-ANN-Model Training Result Statistics----------")    
print("************************************************************")
print('-----------Training the ANN Iris_model... running now ------')
for i in range(1, (epoch + 1)):
    sess.run(optimizer, feed_dict={X_data: X_train, y_target: y_train})
    if i % interval == 0:
        print('Epoch', i, '|', 'Loss:', sess.run(loss, feed_dict={X_data: 
              X_train, y_target: y_train}))
print()
print("----------------------End of Training-----------------------")
print("************************************************************")
print()
#============================================================================
#----------------------------------------------------------------------------
"""
print('Actual Values:', y_test[i], 'Predicted Values:',
Step_14: Define the Prediction here"""
#----------------------------------------------------------------------------
print("-------------------****************-------------------------")
print("-----------Iris-ANN-Model Prediction Statistics-------------") 
print("************************************************************")
print()
print("Actual Values                  Predicted Values")
print("-------------                  ----------------")
for i in range(len(X_test)):
    print(y_test[i], "                ", np.rint(sess.run(
            final_output,feed_dict={X_data: [X_test[i]]})))
print()
print("NB: The Test dataset size used is 10!")
print("----------------End of Prediction Section-------------------")
print("************************************************************")
#============================================================================
"""                          End of Program!                               """
#----------------------------------------------------------------------------
""" Sources: Adopted from many sources in the net"""