#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""======================================================================"""
"""IDE used: Spyder
   Python 3.6.4 on Anaconda3"""
#============================================================================
"""
Created on Wed Apr 25 12:33:13 2018

Due Date: Thursday April 26 23:59 2018

@author: Alem Haddush Fitwi
Email:afitwi1@binghamton.edu
Neural Network & Deep Learning - EECE680C
Department of Electrical & Computer Engineering
Watson Graduate School of Engineering & Applied Science
The State University of New York @ Binghamton
"""
#============================================================================
"""
Assignment-4:Terse Assignment Description:
Design an LSTM that can predict the next value in the generated time series of 
x(n)=sin(0.2*pi*n), where n=0,1,2,3,.... Let each of the training set consitst 
20 samples of x(n). 
"""
#============================================================================
#----------------------------------------------------------------------------
"""
Step_1: Importing Required Packages or keras modules:
"""
#----------------------------------------------------------------------------
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#============================================================================
#----------------------------------------------------------------------------
"""
Step_2: Defining some costant values
"""
#----------------------------------------------------------------------------
num_samples=20 #Size of each training set
in_sin=np.pi*0.2 #  --> pi/5 --> from sin(n*p/5)=sin(in_sin*np.pi)
print("======================================================================")
print("************Preliminary processes and tests before data training******")
print("======================================================================")
#============================================================================
#----------------------------------------------------------------------------
"""
Step_2: Constructing class SineSeries
"""
#----------------------------------------------------------------------------
class SineSeries():
    """
       It generates the time series given number of points, minimum value, &
       maximum value.
    """
    def __init__(self,num_points,xmin,xmax):#arguments to class SineSeries
        self.xmin=xmin
        self.xmax=xmax
        self.num_points=num_points
        self.resolution=(xmax-xmin)/num_points
        self.x_data = np.linspace(xmin,xmax,num_points)
        self.y_true=np.sin(self.x_data*in_sin)
    def ret_true(self, x_series):  #n=x_series
        return np.sin(x_series*in_sin)
    def next_batch(self, batch_size,steps,return_batch_ts=False):
        #Grab a random starting point for each batch
        rand_start=np.random.rand(batch_size,1)
        #convert to be on time series
        ts_start=rand_start*(self.xmax-self.xmin-(steps*self.resolution))
        #create batch time series on the x axis
        batch_ts=ts_start+np.arange(0.0,steps+1)*self.resolution
        #Create Y data for the time series x axis from previus step
        y_batch=np.sin(batch_ts*in_sin)
        #Formatting for RNN
        if return_batch_ts:
            return y_batch[:,:-1].reshape(-1,steps,1), y_batch[:,1:].reshape(-1,steps,1), batch_ts
        else:
            return y_batch[:,:-1].reshape(-1,steps,1), y_batch[:,1:].reshape(-1,steps,1)
#============================================================================
#----------------------------------------------------------------------------
"""
Step_3: Generating the time series
"""
#----------------------------------------------------------------------------
ts_data=SineSeries(300,0,10) #(num_points,xmin,xmax)
plt.figure(1)
plt.title("Time series of sine function")
plt.plot(ts_data.x_data,ts_data.y_true)
y1,y2,ts=ts_data.next_batch(1,num_samples,True)#(batchsize, num_smaple, True)
ts.shape #-->(1,21)
flat=ts.flatten() #shape becomes (21,)
plt.figure(2)
plt.title("Single Training Instance")
plt.plot(flat[1:],y2.flatten(),'*')
plt.figure(3)
plt.title("Testing the Model for single Instance")
plt.plot(ts_data.x_data,ts_data.y_true,label='sin(pi/5*n)')
plt.plot(flat[1:],y2.flatten(),'*',label='Single Training Instance')
plt.legend()
plt.tight_layout()
plt.show()
#============================================================================
#----------------------------------------------------------------------------
"""
Step_4: Generating Training Data
"""
#----------------------------------------------------------------------------
train_inst=np.linspace(5,5 + ts_data.resolution*(num_samples+1),num_samples+1)
train_inst
plt.figure(4)
plt.title('A Training Instance and Target')
plt.plot(train_inst[:-1],ts_data.ret_true(train_inst[:-1]),'bo',markersize=15,alpha=0.5,label='Training Instance')
plt.plot(train_inst[1:],ts_data.ret_true(train_inst[1:]),'ko',markersize=7,label='Target')
plt.legend()
plt.show()
#============================================================================
#----------------------------------------------------------------------------
"""
Step_5: Creating the Model
"""
#----------------------------------------------------------------------------
tf.reset_default_graph()
num_inputs=1
num_neurons=100
num_outputs=1
learning_rate=0.001
num_train_iterations=2200
batch_size=1
#============================================================================
#----------------------------------------------------------------------------
"""
Step_6: Defining Placeholders
"""
#----------------------------------------------------------------------------
X=tf.placeholder(tf.float32,[None,num_samples,num_inputs])
y=tf.placeholder(tf.float32,[None,num_samples,num_outputs])
#============================================================================
#----------------------------------------------------------------------------
"""
Step_7: The RNN-LSTM Cell Layer
"""
#----------------------------------------------------------------------------
cell=tf.contrib.rnn.BasicLSTMCell(num_units=num_neurons, activation=tf.nn.relu)
cell=tf.contrib.rnn.OutputProjectionWrapper(cell,output_size=num_outputs)
outputs,sates=tf.nn.dynamic_rnn(cell,X,dtype=tf.float32)
#============================================================================
#----------------------------------------------------------------------------
"""
Step_8: The Mean Square Error (MSE)
"""
#----------------------------------------------------------------------------
loss =tf.reduce_mean(tf.square(outputs-y))
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
train=optimizer.minimize(loss)
init=tf.global_variables_initializer()
#============================================================================
#----------------------------------------------------------------------------
"""
Step_9: Creating the Session for carrying out training
"""
#----------------------------------------------------------------------------
#======================================================================#
print("======================================================================")
print("...................Training is on progress, please wait for a while!")
print("======================================================================")
saver=tf.train.Saver() #It saves the model for later use
with tf.Session() as sess:
    sess.run(init)
    for iteration in range(num_train_iterations):
        X_batch, y_batch=ts_data.next_batch(batch_size,num_samples)
        sess.run(train,feed_dict={X:X_batch, y:y_batch})
        if iteration % 100==0:
            mse=loss.eval(feed_dict={X:X_batch,y:y_batch})
            print("Iteration\t",iteration, "\tMean Square Error",mse)
    saver.save(sess, "./lstm_series_model_codelog")
#============================================================================
#----------------------------------------------------------------------------
"""
Step_10: Prediction
"""
#----------------------------------------------------------------------------
print("======================================================================")
print("******Prediction and model testing afterconducting data training******")
print("======================================================================")
with tf.Session() as sess:
    saver.restore(sess,"./lstm_series_model_codelog")
    X_new=np.sin(in_sin*(np.array(train_inst[:-1].reshape(-1,num_samples,num_inputs))))
    y_pred=sess.run(outputs,feed_dict={X:X_new})
#============================================================================
#----------------------------------------------------------------------------
"""
Step_11: Testing the model
"""
#----------------------------------------------------------------------------
#Training Instance
plt.figure(5)
plt.title("Testing the Model-after training (prediction)")
plt.plot(train_inst[:-1],np.sin(in_sin*train_inst[:-1]),"bo",markersize=15,alpha=0.5,label='Training Instance')
#----------------------------------------------------------------------------
#Target to predict (correct test values np.sin(in_sin*Train))
plt.plot(train_inst[1:],np.sin(in_sin*train_inst[1:]),"ko",markersize=10,label='Target')
#----------------------------------------------------------------------------
#Models Prediction
plt.plot(train_inst[1:],y_pred[0,:,0],'r.',markersize=10,label='Predictions')
#----------------------------------------------------------------------------
plt.xlabel('Time')
plt.legend()
plt.tight_layout()
plt.show()
print("======================================================================")
print("****************************End of Program****************************")
print("======================================================================")
#============================================================================
#----------------------------------------------------------------------------
"""                            End of program                             """
#----------------------------------------------------------------------------