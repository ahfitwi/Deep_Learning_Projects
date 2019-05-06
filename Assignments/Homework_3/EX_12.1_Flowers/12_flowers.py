#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""======================================================================"""
"""IDE used: Spyder
   Python 3.6.4 on Anaconda3"""
#============================================================================
"""
Created on Mon Apr  2 00:59:13 2018

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
import scipy.io
import scipy.misc
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
#============================================================================
#----------------------------------------------------------------------------
"""
Step_2: Constructing the Class 'InputImages': """        
#----------------------------------------------------------------------------    
class InputImages:
    def extractTestImages(self,Path_1,Path_2,Path_3,X_test):
        self.Path_1=Path_1
        self.Path_2=Path_2
        self.Path_3=Path_3
        self.X_test=X_test
        for flower in range(len(self.X_test)):
            if flower < 10:
                name="yellow"+str(flower)+"."+"jpg"
                image_yellow=self.X_test[flower]  
                scipy.misc.imsave(self.Path_1+name, image_yellow)
            if flower >=10 and flower<20:
                name="white"+str(flower)+"."+"jpg"
                image_yellow=self.X_test[flower]  
                scipy.misc.imsave(self.Path_2+name, image_yellow)
            if flower >= 20 and flower < 30:
                name="pink"+str(flower)+"."+"jpg"
                image_yellow=self.X_test[flower]  
                scipy.misc.imsave(self.Path_3+name, image_yellow)
    #------------------------------------------------------------------------
    def extractTrainingImages(self,Path_4,Path_5,Path_6,X_train):
        self.Path_4=Path_4
        self.Path_5=Path_5
        self.Path_6=Path_6
        self.X_train=X_train
        for flower in range(len(self.X_train)):
            if flower < 70:
                name="yellow"+str(flower)+"."+"jpg"
                image_yellow=self.X_train[flower]  
                scipy.misc.imsave(self.Path_4+name, image_yellow)
            if flower >=70 and flower<140:
                name="white"+str(flower)+"."+"jpg"
                image_yellow=self.X_train[flower]  
                scipy.misc.imsave(self.Path_5+name, image_yellow)
            if flower >= 140 and flower < 210:
                name="pink"+str(flower)+"."+"jpg"
                image_yellow=self.X_train[flower]  
                scipy.misc.imsave(self.Path_6+name, image_yellow)
#============================================================================
#----------------------------------------------------------------------------
"""
Step_3: Constructing the 'Class CNN_Model' using keras: 
"""        
#----------------------------------------------------------------------------       
class CNN_Model:
    def createCNNModel():
        #--------------------------------------------------------------------
        # Part 1 - Building the CNN
        #--------------------------------------------------------------------
        # Initialising the CNN
        classifier = Sequential()
        # Step 1 - Convolution
        classifier.add(Conv2D(32, (3, 3), input_shape = (128, 128, 3), 
                          activation = 'relu'))
        # Step 2 - Pooling
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        # Adding a second convolutional layer
        classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        # Step 3 - Flattening
        classifier.add(Flatten())
        # Step 4 - Full connection
        classifier.add(Dense(units = 128, activation = 'relu'))
        classifier.add(Dense(units = 1, activation = 'sigmoid'))
        # Compiling the CNN
        classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', 
                       metrics = ['accuracy'])
        #--------------------------------------------------------------------
        # Part 2: Fitting the CNN to the images
        #--------------------------------------------------------------------
        train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
        test_datagen = ImageDataGenerator(rescale = 1./255)
        training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (128, 128),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
        test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (128, 128),
                                            batch_size = 32,
                                            class_mode = 'binary')
        classifier.fit_generator(training_set,
                         steps_per_epoch = 210,
                         epochs = 5,
                         validation_data = test_set,
                         validation_steps = 30)
        print("-------------------****************-------------------------")
        print("-------------------Traning is over--------------------------")    
        print("************************************************************")
        #--------------------------------------------------------------------
        # Part_3: Making new predictions
        #--------------------------------------------------------------------
        print("-------------------****************-------------------------")
        print("---------------------Predicting-----------------------------")    
        print("************************************************************")
        test_image = image.load_img('dataset/single_prediction/PorYorW_1.jpg', 
                                    target_size = (128, 128))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = classifier.predict(test_image)
        training_set.class_indices
        if result[0][0] == 0:
            prediction = 'pink'
        if result[0][0] == 1:
            prediction = 'yellow'
        else:
            prediction = 'white'        
        return prediction,result[0][0]
cc=CNN_Model
(p,r)=cc.createCNNModel()
#============================================================================
#----------------------------------------------------------------------------
"""
Step_4: Constructing the 'class Test':                        
""" 
#----------------------------------------------------------------------------
class Test:
    #Reading the innput image which is in .mat format
    im="imagedata.mat"
    dataset= scipy.io.loadmat(im)
    X_test = scipy.io.loadmat(im, variable_names='Xtest').get('Xtest')
    X_train= scipy.io.loadmat(im, variable_names='Xtrain').get('Xtrain')
    Y_test = scipy.io.loadmat(im, variable_names='Ytest').get('Ytest')
    Y_train = scipy.io.loadmat(im, variable_names='Ytrain').get('Ytrain')
    #Extarct test image of each flower type and save them in separet folders
    Path_1="/home/alem/NNDL/dataset/test_set/yellow_flower/"
    Path_2="/home/alem/NNDL/dataset/test_set/white_flower/"
    Path_3="/home/alem/NNDL/dataset/test_set/pink_flower/"
    #Extarct training image of each flower type and save them in d/t folders
    Path_4="/home/alem/NNDL/dataset/training_set/yellow_flower/"
    Path_5="/home/alem/NNDL/dataset/training_set/white_flower/"
    Path_6="/home/alem/NNDL/dataset/training_set/pink_flower/"
    # Instantiating Class InputImages
    class_input_o_1=InputImages()    
    class_input_o_2=InputImages()
    # Instantiating Class CNN_Model    
    class_CNN_o_1=CNN_Model
    #Calling methods of the classes
    class_input_o_1.extractTestImages(Path_1,Path_2,Path_3,X_test)
    class_input_o_2.extractTrainingImages(Path_4,Path_5,Path_6,X_train)
    print("-------------------****************-------------------------")
    print("------------------Traning is running------------------------")    
    print("************************************************************")
    prediction=class_CNN_o_1.createCNNModel()
    print("The predicted image is %s" %(prediction))
    print("----------------Prediction is over--------------------------")
#============================================================================
"""                          End of Program!                               """
#----------------------------------------------------------------------------