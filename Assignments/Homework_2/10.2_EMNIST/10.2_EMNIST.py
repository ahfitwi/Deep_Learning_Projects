#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""======================================================================"""
"""IDE used: Spyder
   Python 3.6.4 on Anaconda3"""
#============================================================================
"""
Created on Sun Feb 25 16:34:39 2018

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
CNN-EMNIST (A ~ Z) Classifier
Goal: To classify Image dataset into 26 classes of letters (A~Z).
The Dataset comprises:
    1) emnist-letters-train-images-idx3-ubyte
    2) emnist-letters-train-labels-idx1-ubyte
    3) emnist-letters-test-images-idx3-ubyte
    4) emnist-letters-test-labels-idx1-ubyte
Dataset Source: https://www.nist.gov/srd/nist-special-database-19
"""
#============================================================================
#----------------------------------------------------------------------------
"""
Step_1: Importing Required Packages or modules:
"""
#----------------------------------------------------------------------------
import struct
import numpy as np
from PIL import Image
import keras
import keras.models
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import model_from_json
from scipy.misc import imread, imresize,imshow
#============================================================================
#----------------------------------------------------------------------------
"""
Step_2: Importing Datasets, Preprocessing, and Training:"
"""

#----------------------------------------------------------------------------
def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

DATA_PATH = 'Data/'
train_img = read_idx(DATA_PATH+'emnist-letters-train-images-idx3-ubyte')
train_label = read_idx(DATA_PATH+'emnist-letters-train-labels-idx1-ubyte')
test_img = read_idx(DATA_PATH+'emnist-letters-test-images-idx3-ubyte')
test_label = read_idx(DATA_PATH+'emnist-letters-test-labels-idx1-ubyte')

#parameter
Number_Classes = 27
batch_size = 64
epochs = 20

# input image dimensions
img_rows, img_cols = 28, 28
depth = 1

#retype data to float 32
train_img = train_img.astype('float32')
test_img = test_img.astype('float32')

#Normalize image data
value_range = 255
train_img /= value_range
test_img /= value_range

#Shape Data
train_img = train_img.reshape(train_img.shape[0], img_rows, img_cols, depth)
test_img = test_img.reshape(test_img.shape[0], img_rows, img_cols, depth )
shape = (img_rows, img_cols, depth )

#One hot encoding
train_label = keras.utils.to_categorical(train_label, Number_Classes )
test_label = keras.utils.to_categorical(test_label, Number_Classes)
#============================================================================
#----------------------------------------------------------------------------
"""
Step_3: Defining and Initializing the Model:
"""
#----------------------------------------------------------------------------
#initialize model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3),activation='relu',
                 input_shape=shape))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(Number_Classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(train_img, train_label,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1, 
    validation_data=(test_img,test_label))

score = model.evaluate(test_img, test_label, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#save the model and serialize it to jason
model_json = model.to_json()
with open("EMmodel.json", "w") as json_file:
          json_file.write(model_json)

#serialize weights to HDF5
model.save_weights("EMmodel.h5")
print("-----------------------------------------------------------------")
print('After a long wait, training is eventually done! Congratulations!')
print("-----------------------------------------------------------------")
#============================================================================
#----------------------------------------------------------------------------
"""
Step_4: Testing and Evaluating the model:
"""
#----------------------------------------------------------------------------
print("-------------Testing using output.png proceeds----------------------")
json_file = open('EMmodel.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
#load weights into new model
loaded_model.load_weights("EMmodel.h5")
print("Loaded Model from disk")

#compile and evaluate loaded model
loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',
                     metrics=['accuracy'])
x = imread('TestImg.png',mode='L')
x = np.invert(x)
x = imresize(x,(28,28))
imshow(x)
x = x.reshape(1,28,28,1)
output = 0
out = loaded_model.predict(x)
id_val = max(out[0])
for i in range(27):
    if out[0][i] == id_val:
        output = i

print('Value is = ', output)
print("List MAX = ", max(out[0]))
print(out)    
print("---------------Testing has come to an end--------------------------")
#============================================================================
#----------------------------------------------------------------------------
"""                           End of Program                              """
#----------------------------------------------------------------------------
""" Dataset Source: https://www.nist.gov/srd/nist-special-database-19
    Code is adopted from various sources in the net"""
