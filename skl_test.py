import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(r'/home/jake/Documents/Programming/Github/Python/')
from SliceOPy import NetSlice, DataSlice
import keras
from sklearn.datasets import make_circles

dataset = make_circles(n_samples=500,noise=0.1,factor=0.5, random_state=1)

features = dataset[0]
labels = dataset[1]

data = DataSlice(Features=features, Labels=labels,Shuffle=True,Split_Ratio=0.7)

layers = []
layers.append(keras.layers.Dense(3,input_dim = 2, activation="sigmoid"))
layers.append(keras.layers.Dense(1, activation="sigmoid"))
model = keras.Sequential(layers)


model = NetSlice(model,'keras',data)

model.compileModel(Optimizer=keras.optimizers.Adam(lr=0.1),Loss='binary_crossentropy')
model.trainModel(Epochs = 100, Verbose=2)


fig = plt.figure()  # create a figure object
ax = fig.add_subplot(1, 2, 1)
ax = model.contourPlot(ax)
ax = fig.add_subplot(1, 2, 2)
ax = model.plotLearningCurve(ax,Plot_Dict={'loss':"Loss",'val_loss':"Test Loss"})
plt.show()