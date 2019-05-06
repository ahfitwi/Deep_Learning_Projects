#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""======================================================================"""
"""IDE used: Spyder
   Python 3.6.4 on Anaconda3"""
#============================================================================
"""
Created on Created on Wed Feb 21 19:20:08 2018

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
Breast-Cancer Classifier!
Task Goal: To Predict class of a given sample 
Every input in the  breast-cancer dataset comprises the following attributes: 
 id,clump_thickness,unif_cell_size,unif_cell_shape,marg_adhesion,
 single_epith_cell_size,bare_nuclei,bland_chrom,norm_nucleoli,mitoses,class
"""
#============================================================================
#----------------------------------------------------------------------------
"""
Step_1: Importing Required Packages or modules:
"""
#----------------------------------------------------------------------------
import tensorflow as tf
import numpy as np
import pandas as pd
#============================================================================
#----------------------------------------------------------------------------
"""
Step_2: Data reading and preprocessing (Dataset must be on the same folder):
"""
#----------------------------------------------------------------------------
data_file_name = 'breast-cancer-wisconsin.data.txt'
data_file_name = 'breast-cancer-wisconsin.data'
fst_lin_1 ="id,clump_thickness,unif_cell_size,unif_cell_shape,marg_adhesion,"
fst_lin_2 = "single_epith_cell_size,bare_nuclei,bland_chrom,norm_nucleoli,"
fst_lin_3="mitoses,class"
first_line=fst_lin_1 + fst_lin_2 + fst_lin_3
with open(data_file_name, "r+") as f:
  content = f.read()
  f.seek(0, 0)
  f.write(first_line.rstrip('\r\n') + '\n' + content)

df = pd.read_csv(data_file_name)

df.replace('?', np.nan, inplace = True)
df.dropna(inplace=True)
df.drop(['id'], axis = 1, inplace = True)

df['class'].replace('2',0, inplace = True)
df['class'].replace('4',1, inplace = True)

df.to_csv("combined_data.csv", index = False)
#============================================================================
#----------------------------------------------------------------------------
"""
Step_3: Accessing and loading the training and test datasets"""
#----------------------------------------------------------------------------
CANCER_TRAINING = "cancer_training.csv"
CANCER_TEST = "cancer_test.csv"
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
               filename=CANCER_TRAINING,target_dtype=np.int,
               features_dtype=np.float32,target_column=-1)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
               filename=CANCER_TEST,target_dtype=np.int,
               features_dtype=np.float32,target_column=-1)
#============================================================================
#----------------------------------------------------------------------------
"""
Step_4: Convert all to real-valued data"""
#----------------------------------------------------------------------------
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=2)]
#============================================================================
#----------------------------------------------------------------------------
"""
Step_5: Building a 3-layer DNN with 10 input, 20 hidden, & 10 output neurons"""
#----------------------------------------------------------------------------
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=2,
                                            model_dir="/tmp/cancer_model")
#============================================================================
#----------------------------------------------------------------------------
"""
Step_5: Fitting the Model"""
#----------------------------------------------------------------------------
classifier.fit(x=training_set.data, 
               y=training_set.target, 
               steps=2000)
#============================================================================
#----------------------------------------------------------------------------
"""
Step_6: Performace or accuracy evaluation"""
#----------------------------------------------------------------------------
accuracy_score = classifier.evaluate(x=test_set.data,
                                     y=test_set.target)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))
#============================================================================
#----------------------------------------------------------------------------
"""
Step_7:Predicting or Classify five new cancer tumor samples."""
#----------------------------------------------------------------------------
def new_samples():
    new_test=np.array([[5, 10, 8, 4, 7, 4, 8, 11, 2],
                       [2, 1, 2, 1, 2, 1, 2, 1, 1],
                       [8, 4, 5, 1, 2, 4, 7, 2, 1],
                       [8, 10, 10, 8, 3, 10, 9, 7, 1],
                       [5, 1, 1, 1, 1, 1, 1, 1, 2]], dtype=np.float32)
    return new_test
new_sample_data=new_samples()
predictions = list(classifier.predict(input_fn=new_samples))
print("-------------------****************-------------------------")
print("-------Five New Cancer Tumor Prediction Statistics----------") 
print("************************************************************")
print(
      "New Samples, Class Predictions:    {}\n"
      .format(predictions))
print("----------------End of Prediction Section-------------------")
print("************************************************************")
#============================================================================
"""                          End of Program!                               """
#----------------------------------------------------------------------------
""" Sources: Adopted from many sources in the net, specifically
    tensor-flow-neural-net-breast-cancer, and I obtained the dataset from the 
    UCI repository """