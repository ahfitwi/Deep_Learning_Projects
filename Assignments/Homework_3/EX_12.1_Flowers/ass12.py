#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 02:31:39 2018

@author: alem
"""

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
It comprises four classes, namely
  1) Class InputImages: handles preliminary image iputing & processing
  2) Class CNN_Model: creates the Input, Hidden and output layers for training
  3) Class Prediction: makes new predictions
  4) Class Testing: handles the calling and testing of all other classes
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
import numpy as np
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
import scipy.io
import scipy.misc
from keras.layers import AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
#============================================================================
#----------------------------------------------------------------------------
"""
Step_2: Constructing the Class 'InputImages': """        
#----------------------------------------------------------------------------    
train_data = scipy.io.loadmat('imagedata.mat')
X_test = scipy.io.loadmat('imagedata.mat', variable_names='Xtest').get('Xtest')
X_train= scipy.io.loadmat('imagedata.mat', variable_names='Xtrain').get('Xtrain')
Y_test = scipy.io.loadmat('imagedata.mat', variable_names='Ytest').get('Ytest')
Y_train = scipy.io.loadmat('imagedata.mat', variable_names='Ytrain').get('Ytrain')

Path_1="/home/alem/NNDL/dataset/test_set/yellow_flower/"
Path_2="/home/alem/NNDL/dataset/test_set/white_flower/"
Path_3="/home/alem/NNDL/dataset/test_set/pink_flower/"
Path_4="/home/alem/NNDL/dataset/training_set/yellow_flower/"
Path_5="/home/alem/NNDL/dataset/training_set/white_flower/"
Path_6="/home/alem/NNDL/dataset/training_set/pink_flower/"

for flower in range(len(X_train)):
    if flower < 70:
        name="yellow"+str(flower)+"."+"jpg"
        image_yellow=X_train[flower]  
        scipy.misc.imsave(Path_4+name, image_yellow)
    if flower >=70 and flower<140:
        name="white"+str(flower)+"."+"jpg"
        image_yellow=X_train[flower]  
        scipy.misc.imsave(Path_5+name, image_yellow)
    if flower >= 140 and flower < 210:
        name="pink"+str(flower)+"."+"jpg"
        image_yellow=X_train[flower]  
        scipy.misc.imsave(Path_6+name, image_yellow)   

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))

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
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

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
                         epochs = 1,
                         validation_data = test_set,
                         validation_steps = 30)

# Part 3 - Making new predictions
test_image = image.load_img("/home/alem/NNDL/tensorflow-101-master/dataset/monalisa.png", 
                            target_size = (128, 128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
    print(prediction)
else:
    prediction = 'cat'
    print(prediction)

#------------------------------
#function for drawing bar chart for emotion preditions
def emotion_analysis(emotions):
    objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    y_pos = np.arange(len(objects))
    
    plt.bar(y_pos, emotions, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('percentage')
    plt.title('emotion')
    
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
			testing_img = testing_img.reshape([48, 48]);
			
			plt.gray()
			plt.imshow(testing_img)
			plt.show()
			
			print(i)
			
			emotion_analysis(i)
			print("----------------------------------------------")
		index = index + 1

#------------------------------
#make prediction for custom image out of test set

img = image.load_img("C:/Users/IS96273/Desktop/jackman.png", grayscale=True, target_size=(48, 48))

x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)

x /= 255

custom = model.predict(x)
emotion_analysis(custom[0])

x = np.array(x, 'float32')
x = x.reshape([48, 48]);

plt.gray()
plt.imshow(x)
plt.show()
