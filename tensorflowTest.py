# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 16:14:53 2018

@author: lhe39759
"""

import keras
import tensorflow as tf
from sklearn.datasets import make_moons, make_circles
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append(r'C:\Users\lhe39759\Documents\GitHub\CCPi-ML/')
from SliceOPy import NetSlice, DataSlice
#dataset = make_moons(n_samples=300,noise=0.20, random_state=1)
dataset = make_circles(n_samples=300,noise=0.20, factor=0.1,random_state=1)

features = dataset[0]
labels = dataset[1]

features[:,0] = (features[:,0]+1.5)/3.0
features[:,1] = (features[:,1]+1.5)/3.0

x1_min = np.amin(features[:,0])
x1_max = np.amax(features[:,0])
x2_min = np.amin(features[:,1])
x2_max = np.amax(features[:,1])

plt.scatter(features[:,0],features[:,1],edgecolor="black",linewidth=1,c=labels)
plt.xlabel("x1")
plt.ylabel("x2")
plt.colorbar()
plt.show()

netData = DataSlice.DataSlice(Features = features, Labels= labels,Shuffle=True)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def buildModel():
    x = tf.placeholder(tf.float32, shape=[None, 2])
    y_ = tf.placeholder(tf.float32, shape=[None, 1])
    
    
    x = tf.placeholder(tf.float32, [None, 2])
    
    ####################
    #Dense
    ####################
    W_1 = weight_variable([2, 4])
    b_1= bias_variable([4])
    
    h_1 = tf.nn.tanh(tf.matmul(x, W_1) + b_1)
    
    W_2 = weight_variable([4, 3])
    b_2= bias_variable([3])
    
    h_2 = tf.nn.tanh(tf.matmul(h_1, W_2) + b_2)
    
    
    ###################
    #Output
    ####################
    W_Out = weight_variable([3, 1])
    b_Out = bias_variable([1])
    
    y_conv = tf.matmul(h_2, W_Out) + b_Out
    
    
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    return train_step, cross_entropy,  correct_prediction, accuracy,x,y_


model = NetSlice.NetSlice(buildModel(),'tensorflow',netData)
model.compileModel(Model=buildModel)
model.trainModel(Epochs=10,Batch_size=10,Verbose=2)