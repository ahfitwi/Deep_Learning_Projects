#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""======================================================================"""
"""IDE used: Spyder
   Python 3.6.4 on Anaconda3"""
#============================================================================
"""
Created on Mon Apr  2 00:59:13 2018

Due Date: Friday Feb 09 23:59 2018

@author: Alem Haddush Fitwi
Email:afitwi1@binghamton.edu
Neural Network & Deep Learning - EECE680C
Department of Electrical & Computer Engineering
Watson Graduate School of Engineering & Applied Science
The State University of New York @ Binghamton
"""
#============================================================================
"""
Terse Assignment Description:
Design a CNN classifier for flower classification. The data sets consists of 3
types of lowers, each has 80 images. 70 images per flower are training, and 10
 images per flower are testing.
"""
#============================================================================
"""
Step_0: Laconic Description of the solution Program Organization
It comprises three classes, namely
  1) Class InputImages: handles preliminary image iputing & processing
  2) Class CNN_Model: creates the CNN training models & Makes prediction
  3) Class Testing: handles the calling and testing of all other classes
"""
#============================================================================
#----------------------------------------------------------------------------
"""
Step_1: Importing Required Packages or keras modules:
"""
#----------------------------------------------------------------------------
import scipy.misc
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
#variables
num_classes = 3 #pink, yellow, and white
batch_size = 32
epochs = 1
#----------------------------------------------------------------------------
#initialize trainset and test set
im="imagedata.mat"
dataset= scipy.io.loadmat(im)
x_test = scipy.io.loadmat(im, variable_names='Xtest').get('Xtest')
x_train= scipy.io.loadmat(im, variable_names='Xtrain').get('Xtrain')
y_test = scipy.io.loadmat(im, variable_names='Ytest').get('Ytest')
y_train = scipy.io.loadmat(im, variable_names='Ytrain').get('Ytrain')
#----------------------------------------------------------------------------
#data transformation for train and test sets
x_train = np.array(x_train, 'float32')
y_train = np.array(y_train, 'float32')
x_test = np.array(x_test, 'float32')
y_test = np.array(y_test, 'float32')

x_train /= 255 #normalize inputs between [0, 1]
x_test /= 255

x_train = x_train.reshape(x_train.shape[0], 128, 128, 3)
x_train = x_train.astype('float32')
x_test = x_test.reshape(x_test.shape[0], 128, 128, 3)
x_test = x_test.astype('float32')

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
#----------------------------------------------------------------------------
#construct CNN structure
model = Sequential()

#1st convolution layer
model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(128,128,3)))
model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))

#2nd convolution layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

#3rd convolution layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

model.add(Flatten())

#fully connected neural networks
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(num_classes, activation='softmax'))
#------------------------------
#batch process
gen = ImageDataGenerator()
train_generator = gen.flow(x_train, y_train, batch_size=batch_size)

#------------------------------

model.compile(loss='categorical_crossentropy'
    , optimizer=keras.optimizers.Adam()
    , metrics=['accuracy']
)

#------------------------------

fit = True

if fit == True:
	#model.fit_generator(x_train, y_train, epochs=epochs) #train for all trainset
	model.fit_generator(train_generator, steps_per_epoch=batch_size, epochs=epochs) #train for randomly selected one
else:
	model.load_weights('/data/facial_expression_model_weights.h5') #load weights
	
#------------------------------
#overall evaluation
score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', 100*score[1])

#------------------------------
#function for drawing bar chart for emotion preditions
def flower_analysis(flowers):
    objects = ('pink', 'yellow', 'White')
    y_pos = np.arange(len(objects))
    
    plt.bar(y_pos, flowers, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('percentage')
    plt.title('flower')
    
    plt.show()
#------------------------------

monitor_testset_results = False

if monitor_testset_results == True:
	#make predictions for test set
	predictions = model.predict(x_test)

	index = 0
	for i in predictions:
		if index < 30 and index >= 20:
			#print(i) #predicted scores
			#print(y_test[index]) #actual scores
			
			testing_img = np.array(x_test[index], 'float32')
			testing_img = testing_img.reshape([128, 128]);
			
			plt.gray()
			plt.imshow(testing_img)
			plt.show()
			
			print(i)
			
			flower_analysis(i)
			print("----------------------------------------------")
		index = index + 1

#------------------------------
#make prediction for custom image out of test set

img = image.load_img("/home/alem/NNDL/white15.jpg", grayscale=True, target_size=(128, 128,3))

x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)

x /= 255

custom = model.predict(x)
flower_analysis(custom[0])

x = np.array(x, 'float32')
x = x.reshape([128, 128]);

plt.gray()
plt.imshow(x)
plt.show()
#------------------------------
#============================================================================
"""                          End of Program!                               """
#----------------------------------------------------------------------------