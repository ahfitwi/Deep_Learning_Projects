#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""======================================================================"""
"""IDE used: Spyder
   Python 3.6.4 on Anaconda3"""
#============================================================================
"""
Created on Tue Apr  3 22:54:34 2018

Due Date: Thursday April 05 23:59 2018

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
Design a CNN classifier for facial expression classification. The dataset 
comprises 28K (28709)training and 3K (3589) testing images. Each image was 
stored as 48×48 pixel. The pure dataset consists of image pixels (48×48=2304 
values), emotion of each image and usage type (as train/validation instance).
There are 7 different emotions or facial expressions: angry, disgust, fear, 
happy, sad, surprise, and neutral which are encoded as 0, 1, 2, 3, 5, and 6. 
This is represented using a num_classes!
"""
#============================================================================
#----------------------------------------------------------------------------
"""
Step_1: Importing Required Packages or keras modules:
"""
#----------------------------------------------------------------------------
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
#============================================================================
#----------------------------------------------------------------------------
"""
Step_2: Pre-processing the input dataset in facial expressions folder """        
#----------------------------------------------------------------------------
class InputImageProcessing:
    def loadAndProcessDataset(self,x_train,y_train,x_test,y_test):              
        #data transformation for train and test sets
        self.x_train=x_train
        self.y_train=y_train
        self.x_test=x_test
        self.y_test=y_test
        x_train = np.array(x_train, 'float32')
        y_train = np.array(y_train, 'float32')
        x_test = np.array(x_test, 'float32')
        y_test = np.array(y_test, 'float32')
        #normalize inputs between [0, 1]
        x_train /= 255 
        x_test /= 255
        x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
        x_train = x_train.astype('float32')
        x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)
        x_test = x_test.astype('float32')
        return x_train,y_train,x_test,y_test
#----------------------------------------------------------------------------    
class ProcessedDataset:
    #Defining important variables
    num_classes = 7 
    batch_size = 256
    epochs = 5
    #Loading the required datasets, training and validation 
    x_train1 = np.load('X_train.npy')
    y_train1 = np.load('Y_train.npy')
    x_test1 = np.load('X_val.npy')
    y_test1 = np.load('Y_val.npy')      
    # Instantiating Class InputImages
    class_o_1=InputImageProcessing()     
    (x_train,y_train,x_test,y_test)=class_o_1.loadAndProcessDataset(x_train1,
                                      y_train1,x_test1,y_test1)
#============================================================================
#----------------------------------------------------------------------------
"""
Step_3: Constructing the CNN Model using keras:    
"""        
#----------------------------------------------------------------------------
class CNN_Model:   
    #Calling class ProcessedDataset
    O_2=ProcessedDataset;
    #Defining important variables
    num_classes = 7 
    batch_size = 256
    epochs = 5
    #Accessing the training and testing datasets
    class_ob_1=InputImageProcessing()   
    #construct CNN structure
    model = Sequential()
    #1st convolution layer
    model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48,48,1)))
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
    #-------------------------------------------------------------------------
    #Batch processing
    gen = ImageDataGenerator()
    train_generator = gen.flow(O_2.x_train, O_2.y_train, batch_size=batch_size)
    model.compile(loss='categorical_crossentropy'
              , optimizer=keras.optimizers.Adam()
              , metrics=['accuracy'])
    #Fitting the CNN to the images
    model.fit_generator(train_generator, steps_per_epoch=batch_size, 
                            epochs=epochs)
    #-------------------------------------------------------------------------
    #Overall evaluation
    score = model.evaluate(O_2.x_test, O_2.y_test)
    print("-------------------************-------------------------")
    print('***************Test loss:', score[0],"******************")
    print('***************Test accuracy:', 100*score[1],"***************")
    print("-------------------************-------------------------")
    #-------------------------------------------------------------------------

#============================================================================
#----------------------------------------------------------------------------
"""
Step_4: Drawing bar chart for the facial expression/emotion preditions
"""        
#---------------------------------------------------------------------------- 
class Performance_Prediction:
    O_3=CNN_Model
    O_4=ProcessedDataset;
    def emotion_analysis(emotions):
        objects = ('angry','disgust','fear','happy','sad','surprise','neutral')
        y_pos = np.arange(len(objects))    
        plt.bar(y_pos, emotions, align='center', alpha=0.5)
        plt.xticks(y_pos, objects)
        plt.ylabel('percentage')
        plt.title('emotion')    
        plt.show()
    #------------------------------------------------------------------------- 
    monitor_testset_results = False
    if monitor_testset_results == True:
        #make predictions for test set
        predictions = O_3.model.predict(O_4.x_test)
        index = 0
        for i in predictions:
            if index < 30 and index >= 20:	
                testing_img = np.array(O_4.x_test[index], 'float32')
                testing_img = testing_img.reshape([48, 48]);
                plt.gray()
                plt.imshow(testing_img)
                plt.show()
                print(i)
                emotion_analysis(i)
                print("----------------------------------------------")
                index = index + 1
    #------------------------------------------------------------------------- 
#============================================================================
#----------------------------------------------------------------------------
"""
Step_5: Making prediction for a custom image out from the test set
"""        
#----------------------------------------------------------------------------
class CustomImage:
    O_5=CNN_Model
    O_6=Performance_Prediction
    img = image.load_img("dataset/jackman.png",grayscale=True, 
                         target_size=(48, 48))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    x /= 255
    custom = O_5.model.predict(x)
    O_6.emotion_analysis(custom[0])
    x = np.array(x, 'float32')
    x = x.reshape([48, 48]);
    plt.gray()
    plt.imshow(x)
    plt.show()
#============================================================================
#----------------------------------------------------------------------------
"""
Step_6: The Test class
"""        
#----------------------------------------------------------------------------

class Test:    
    O_7=CNN_Model
    O_8=Performance_Prediction
    O_9=CustomImage     
#============================================================================
"""                          End of Program!                               """
#----------------------------------------------------------------------------