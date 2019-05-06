#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""======================================================================"""
"""IDE used: Spyder
   Python 3.6.4 on Anaconda3"""
#============================================================================
"""
Created on Wed April 25 18:56:08 2018

Due Date: on Wed May 17 23:59 2018

@authors: Alem, Charlie, & Xiaojing Xia
Email:afitwi1@binghamton.edu
Neural Network & Deep Learning - EECE680C
Department of Electrical & Computer Engineering
Watson Graduate School of Engineering & Applied Science
The State University of New York @ Binghamton
"""
#============================================================================
"""
ANN-CNN OCT Image Classifier
Goal: To classify Retina OCT Image dataset into 4 classes using namely:
                0  --> Normal
                1  --> CNV
                2  --> DME
                3  --> DRUSEN
"""
#============================================================================
#----------------------------------------------------------------------------
"""
Step_1: Importing Required Modules
"""
#----------------------------------------------------------------------------
import numpy as np
import cv2
import os
import skimage
import skimage.transform
from tqdm import tqdm
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from time import time
import urllib.request as urllib
import zipfile
import sys
from pdb import set_trace
#============================================================================
#----------------------------------------------------------------------------
"""
Step_2: Input Dataset Loading and preprocessing
"""
#----------------------------------------------------------------------------
"""We used Google Collabatory to run our network
The dataset must be obtained over the internet since we can't put files on 
Google's machines.
We hosted the data on a cloud-9 server which does NOT run all the time
load zipped data from the internet"""
#----------------------------------------------------------------------------
response = urllib.urlopen('https://neural-data-host-charliedmiller.c9users.io/OCT_2017_Dataset.zip')
print("Got zip data -- Writing zip data")
#Write data to a file - Google will delete this when the program terminates
data = response.read()
with open("OCT_2017_Dataset.zip",'wb') as df:
  df.write(data)
#----------------------------------------------------------------------------
#Releive the machine of RAM
data=""
print("Wrote data, now unzipping")
#Unzip the file
zip_ref = zipfile.ZipFile("OCT_2017_Dataset.zip", 'r')
zip_ref.extractall()
zip_ref.close()
print("Done extracting, now loading real data in")
#----------------------------------------------------------------------------
#Image resizing size
imageSize=256
#Folders from the zipped folder
train_dir = "OCT_2017_Dataset/train/"
test_dir =  "OCT_2017_Dataset/test/"
# Labels ==> ['DME', 'CNV', 'NORMAL', 'DRUSEN']

#Load Training and Testing Datasets from the newly created folders
def get_data(folder):
    """
    Load the data and labels from the given folder.
    """
    X = []
    y = []
    for folderName in os.listdir(folder):
        if not folderName.startswith('.'):
            if folderName in ['NORMAL']:
                label = 0
            elif folderName in ['CNV']:
                label = 1
            elif folderName in ['DME']:
                label = 2
            elif folderName in ['DRUSEN']:
                label = 3            
            for image_filename in tqdm(os.listdir(folder + folderName)):
                img_file = cv2.imread(folder + folderName + '/' + 
                                      image_filename)
                if img_file is not None:
                    img_file = skimage.transform.resize(img_file, 
                                            (imageSize, imageSize, 1))
                    img_arr = np.asarray(img_file)
                    X.append(img_arr)
                    y.append(label)
    X = np.asarray(X)
    y = np.asarray(y)
    return X,y
  
#Do the loading into varibles  
print("Please wait ... Traing dataset images are being extracted ...")
train_x, train_y = get_data(train_dir) 
print()
print("Please wait ... Testing dataset images are being extracted ...")
test_x, test_y= get_data(test_dir)
# Encode labels to hot vectors (ex : DME -> [0,0,1,0])
from keras.utils.np_utils import to_categorical
train_y = to_categorical(train_y, num_classes = 4)
test_y = to_categorical(test_y, num_classes = 4)
#Labels
train_l=['NORMAL', 'CNV', 'DME', 'DRUSEN']
test_l=['NORMAL', 'CNV', 'DME', 'DRUSEN']
#============================================================================
#----------------------------------------------------------------------------
"""
Step_3: Defining and Building the CNN Model:
"""
#----------------------------------------------------------------------------
def model():
    _IMAGE_SIZE = 256
    _IMAGE_CHANNELS = 1
    _NUM_CLASSES = 4
    _RESHAPE_SIZE = 4*4*128
    #----------------------------------------------------------------------
    #Initialize input placeholders
    with tf.name_scope('data'):
        x = tf.placeholder(tf.float32, shape=[None, 
                   _IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_CHANNELS], name='Input')
        y = tf.placeholder(tf.float32, shape=[None, _NUM_CLASSES], 
                           name='Output')
        #Input should be _IMAGE_SIZE x _IMAGE_SIZE
        x_image = tf.reshape(x, [-1, _IMAGE_SIZE, _IMAGE_SIZE, 
                                 _IMAGE_CHANNELS], name='images')
    #----------------------------------------------------------------------
    #function to create variables (weights, biases) with 
    #some probably to be dropped
    def variable_with_weight_decay(name, shape, stddev, wd):
        dtype = tf.float32
        var = variable_on_cpu( name, shape, tf.truncated_normal_initializer(
                                                  stddev=stddev, dtype=dtype))
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, 
                                       name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var
    #----------------------------------------------------------------------
    def variable_on_cpu(name, shape, initializer):
        with tf.device('/cpu:0'):
            dtype = tf.float32
            var = tf.get_variable(name, shape, initializer=initializer, 
                                  dtype=dtype)
        return var
    #----------------------------------------------------------------------
    #Convolution Layer_1
    with tf.variable_scope('conv1') as scope:
        #Create kernel - 5x5 size, 64 outpupts
        kernel = variable_with_weight_decay('weights', shape=[5, 5, 1, 64], 
                                            stddev=5e-2, wd=0.0)
        #Do the convolution
        conv = tf.nn.conv2d(x_image, kernel, [1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        #Compute output of layer
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
    
    #Log results 
    tf.summary.histogram('Convolution_layers/conv1', conv1)
    tf.summary.scalar('Convolution_layers/conv1', tf.nn.zero_fraction(conv1))
    
    #normalize the data
    norm1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, 
                                                           name='norm1')
    #----------------------------------------------------------------------
    #Max_pool_1
    pool1 = tf.nn.max_pool(norm1, ksize=[1, 3, 3, 1], strides=[1, 4, 4, 1], 
                                            padding='SAME', name='pool1')
    #----------------------------------------------------------------------
    #Convolution Layer_2 - 5x5, output 64 channels
    with tf.variable_scope('conv2') as scope:
        kernel = variable_with_weight_decay('weights', shape=[5, 5, 64, 64], 
                                            stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
    tf.summary.histogram('Convolution_layers/conv2', conv2)
    tf.summary.scalar('Convolution_layers/conv2', tf.nn.zero_fraction(conv2))

    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, 
                      name='norm2')
    #----------------------------------------------------------------------
    #Max_pool_2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 4, 4, 1], 
                           padding='SAME', name='pool2')
    #----------------------------------------------------------------------
    #Convolution Layer_3, 3x3,128 output channels
    #We will not max-pool the next few layers since the images are
    #sufficiently small to process
    with tf.variable_scope('conv3') as scope:
        kernel = variable_with_weight_decay('weights', shape=[3, 3, 64, 128], 
                                            stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(pre_activation, name=scope.name)
    tf.summary.histogram('Convolution_layers/conv3', conv3)
    tf.summary.scalar('Convolution_layers/conv3', tf.nn.zero_fraction(conv3))    
    #----------------------------------------------------------------------
    #Convolution Layer_4, 3x3, 128 output channels, no max-pool
    with tf.variable_scope('conv4') as scope:
        kernel = variable_with_weight_decay('weights', shape=[3, 3, 128, 128], 
                                            stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(pre_activation, name=scope.name)
    tf.summary.histogram('Convolution_layers/conv4', conv4)
    tf.summary.scalar('Convolution_layers/conv4', tf.nn.zero_fraction(conv4))
    #----------------------------------------------------------------------
    #Convolution Layer_5,3x3, 128 output channels, WITH max-pool
    with tf.variable_scope('conv5') as scope:
        kernel = variable_with_weight_decay('weights', shape=[3, 3, 128, 128], 
                                            stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(pre_activation, name=scope.name)
    tf.summary.histogram('Convolution_layers/conv5', conv5)
    tf.summary.scalar('Convolution_layers/conv5', tf.nn.zero_fraction(conv5))

    norm3 = tf.nn.lrn(conv5, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, 
                      name='norm3')
    #----------------------------------------------------------------------
    #Max_pool_3
    pool3 = tf.nn.max_pool(norm3, ksize=[1, 3, 3, 1], strides=[1, 4, 4, 1], 
                           padding='SAME', name='pool3')
    #----------------------------------------------------------------------
    #Hold 2 fully connected layers - more depth
    #Fully Connected Layer_1 - 384 outputs
    with tf.variable_scope('fully_connected1') as scope:
        #reshape the convolution layer outputs to a flat vector
        reshape = tf.reshape(pool3, [-1, _RESHAPE_SIZE])
        dim = reshape.get_shape()[1].value
        #Get weights and biases
        weights = variable_with_weight_decay('weights', shape=[dim, 384], 
                                             stddev=0.04, wd=0.004)
        biases = variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        #Apply non-linear function (Rectified Linear Unit)
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, 
                            name=scope.name)
    #Log results    
    tf.summary.histogram('Fully connected layers/fc1', local3)
    tf.summary.scalar('Fully connected layers/fc1', 
                      tf.nn.zero_fraction(local3))
    #----------------------------------------------------------------------
    #Fully Connected Layer_2 - 192 outputs
    with tf.variable_scope('fully_connected2') as scope:
        weights = variable_with_weight_decay('weights', shape=[384, 192], 
                                             stddev=0.04, wd=0.004)
        biases = variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, 
                            name=scope.name)
    tf.summary.histogram('Fully connected layers/fc2', local4)
    tf.summary.scalar('Fully connected layers/fc2', 
                      tf.nn.zero_fraction(local4))
    #----------------------------------------------------------------------
    #Output layer - condensce FC layer to 4 output neurons
    #One for each class
    with tf.variable_scope('output') as scope:
        weights = variable_with_weight_decay('weights', [192, _NUM_CLASSES], 
                                             stddev=1 / 192.0, wd=0.0)
        biases = variable_on_cpu('biases', [_NUM_CLASSES], 
                                 tf.constant_initializer(0.0))
        
        #Put output layer through a softmax layer
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, 
                                name=scope.name)
    #----------------------------------------------------------------------
    tf.summary.histogram('Fully connected layers/output', softmax_linear)

    global_step = tf.Variable(initial_value=0, name='global_step', 
                              trainable=False)
    #State which class the neural network thinks the 
    #image looks like the most
    y_pred_cls = tf.argmax(softmax_linear, axis=1)

    return x, y, softmax_linear, global_step, y_pred_cls

#============================================================================
#----------------------------------------------------------------------------
"""
Step_4: Defining the CNN Training Module:"""
#----------------------------------------------------------------------------
x, y, output, global_step, y_pred_cls = model()

_IMG_SIZE = 256
_NUM_CHANNELS = 1
_BATCH_SIZE = 50
_CLASS_SIZE = 4
_ITERATION = 10000
_SAVE_PATH = "./tensorboard/rectinaOCT/"

#Compute loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, 
                                                              labels=y))
#Optimize with RMSPropagation Optimization
optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-3).minimize(loss, 
                                     global_step=global_step)

#Do the correcting
correct_prediction = tf.equal(y_pred_cls, tf.argmax(y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar("Accuracy/train", accuracy)


#Start progress saver and tensorflow session
merged = tf.summary.merge_all()
saver = tf.train.Saver()
sess = tf.Session()
train_writer = tf.summary.FileWriter(_SAVE_PATH, sess.graph)

#----------------------------------------------------------------------------
#Attempt to restore a previous session
try:
    print("Trying to restore last checkpoint ...")
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=_SAVE_PATH)
    saver.restore(sess, save_path=last_chk_path)
    print("Restored checkpoint from:", last_chk_path)
except:
    #Create a new session if there was no previous
    print("Failed to restore checkpoint. Initializing variables instead.")
    sess.run(tf.global_variables_initializer())
#----------------------------------------------------------------------------

def train(num_iterations):
    '''
        Train CNN
    '''
    for i in range(num_iterations):
        #State Iteration number
        sys.stdout.flush()
        sys.stdout.write("\rIteration %d" % i)
        
        #Select a random batch to train
        randidx = np.random.randint(len(train_x), size=_BATCH_SIZE)
        batch_xs = train_x[randidx]
        batch_ys = train_y[randidx]

        #Do the training
        start_time = time()
        i_global, _ = sess.run([global_step, optimizer], 
                               feed_dict={x: batch_xs, y: batch_ys})
        duration = time() - start_time

        #State loss and batch accuracy every 10 iterations
        if (i_global % 10 == 0) or (i == num_iterations - 1):
            _loss, batch_acc = sess.run([loss, accuracy], 
                                        feed_dict={x: batch_xs, y: batch_ys})
            msg1="Global Step: {0:>6}, accuracy: {1:>6.1%}, "
            msg2="loss = {2:.2f} ({3:.1f} examples/sec, {4:.2f} sec/batch)"
            msg = msg1+msg2
            print(msg.format(i_global, batch_acc, _loss, 
                             _BATCH_SIZE / duration, duration))

        #Perform a test for accuracy on unseen examples
        if (i_global % 100 == 0) or (i == num_iterations - 1):
            
            data_merged, global_1 = sess.run([merged, global_step], 
                                         feed_dict={x: batch_xs, y: batch_ys})
            
            #This is is the acutal test function
            acc = predict_test()

            #Log the results
            summary = tf.Summary(value=[
                tf.Summary.Value(tag="Accuracy/test", simple_value=acc),
            ])
            train_writer.add_summary(data_merged, global_1)
            train_writer.add_summary(summary, global_1)

            #Save progress up to here
            saver.save(sess, save_path=_SAVE_PATH, global_step=global_step)
            print("Saved checkpoint.")
#----------------------------------------------------------------------------
#actual testing function
def predict_test(show_confusion_matrix=False):
    '''
        Make prediction for all images in test_x
    '''
    i = 0
    predicted_class = np.zeros(shape=len(test_x), dtype=np.int)
    
    while i < len(test_x):
        #Create a test batch
        j = min(i + _BATCH_SIZE, len(test_x))
        batch_xs = test_x[i:j, :]
        batch_ys = test_y[i:j, :]
        #Do the test
        predicted_class[i:j] = sess.run(y_pred_cls, feed_dict={x: batch_xs, 
                       y: batch_ys})
        i = j

    #Get results of the test
    correct = (np.argmax(test_y, axis=1) == predicted_class)
    acc = correct.mean()*100
    correct_numbers = correct.sum()
    #print results of the test
    print("Accuracy on Test-Set: {0:.2f}% ({1} / {2})".format(acc, 
          correct_numbers, len(test_x)))

    if show_confusion_matrix is True:
        cm = confusion_matrix(y_true=np.argmax(test_y, axis=1), 
                              y_pred=predicted_class)
        for i in range(_CLASS_SIZE):
            class_name = "({}) {}".format(i, test_l[i])
            print(cm[i, :], class_name)
        class_numbers = [" ({0})".format(i) for i in range(_CLASS_SIZE)]
        print("".join(class_numbers))

    return acc
#----------------------------------------------------------------------------

if _ITERATION != 0:
    train(_ITERATION)

#============================================================================
#----------------------------------------------------------------------------
"""
Step_5: The prediction modules:
"""
#----------------------------------------------------------------------------
try:
    print("Trying to restore last checkpoint ...")
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=_SAVE_PATH)
    saver.restore(sess, save_path=last_chk_path)
    print("Restored checkpoint from:", last_chk_path)
except:
    print("Failed to restore checkpoint. Initializing variables instead.")
    sess.run(tf.global_variables_initializer())
#----------------------------------------------------------------------------
#The last testing function
i = 0
predicted_class = np.zeros(shape=len(test_x), dtype=np.int)
while i < len(test_x):
    j = min(i + _BATCH_SIZE, len(test_x))
    batch_xs = test_x[i:j, :]
    batch_ys = test_y[i:j, :]
    predicted_class[i:j] = sess.run(y_pred_cls, feed_dict={x: batch_xs, 
                   y: batch_ys})
    i = j
#----------------------------------------------------------------------------
correct = (np.argmax(test_y, axis=1) == predicted_class)
acc = correct.mean()*100
correct_numbers = correct.sum()
print("Accuracy on Test-Set: {0:.2f}% ({1} / {2})".format(acc, correct_numbers, 
      len(test_x)))
#----------------------------------------------------------------------------
cm = confusion_matrix(y_true=np.argmax(test_y, axis=1), y_pred=predicted_class)
for i in range(_CLASS_SIZE):
    class_name = "({}) {}".format(i, test_l[i])
    print(cm[i, :], class_name)
class_numbers = [" ({0})".format(i) for i in range(_CLASS_SIZE)]
print("".join(class_numbers))
#----------------------------------------------------------------------------

sess.close()

#============================================================================
"""                          End of Program!                               """
#----------------------------------------------------------------------------
"""References: for some special operations we referred to 
https://www.kaggle.com/kmader/detect-retina-damage-from-oct-images-hr and other
Internet sources. Refer to the list of references at the end of our report"""

