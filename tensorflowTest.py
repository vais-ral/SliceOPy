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
import math
sys.path.append(r'C:\Users\lhe39759\Documents\GitHub/')
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
#
#plt.scatter(features[:,0],features[:,1],edgecolor="black",linewidth=1,c=labels)
#plt.xlabel("x1")
#plt.ylabel("x2")
#plt.colorbar()
#plt.show()

netData = DataSlice.DataSlice(Features = features, Labels= labels,Shuffle=True)
#netData.y_train = netData.y_train.reshape(netData.y_train.shape[0],1)
#netData.y_test = netData.y_test.reshape(netData.y_test.shape[0],1)
netData.oneHot(2)
print(netData.y_train.shape)
 
def weight_variable(shape):
  initial = tf.random_normal(shape)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def buildModel():
    
    x = tf.placeholder(tf.float32, shape=[None, 2])
    y_ = tf.placeholder(tf.float32, shape=[None, 2])
    
   # is_training=tf.Variable(True,dtype=tf.bool)

    initializer = tf.contrib.layers.xavier_initializer()
    h0 = tf.layers.dense(x, 6, activation=tf.nn.sigmoid, kernel_initializer=initializer)
    # h0 = tf.nn.dropout(h0, 0.95)
    predicted = tf.layers.dense(h0, 2, activation=tf.nn.sigmoid)
    

    return x,y_,predicted,[]

def Loss(out,pred,*kwargs):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=out, logits=pred)
    cost = tf.reduce_mean(cross_entropy)
    return cost

def Optimizer(cost,*kwargs):

    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)   
    return optimizer

def Accuracy(lab,pred,*kwargs): 
    
    correct_pred = tf.equal(tf.round(pred), lab)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    return accuracy


#op, ls, accuracy,g ,h = buildModel()

model = NetSlice.NetSlice(buildModel,'tensorflow',netData)
model.compileModel(Optimizer = Optimizer, Loss = Loss, Metrics = Accuracy)
model.trainModel(Epochs=1000,Batch_size=10,Verbose=2)