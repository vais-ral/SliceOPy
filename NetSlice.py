# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 10:19:08 2018

@author: lhe39759
"""
#import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
import sys
import os
sys.path.append(r'$HOME\Documents/Programming/Python/CCPi-ML/')
from SliceOPy import  DataSlice
import pickle

class NetSlice:

#""" Model Building
#    Choice of backends are: 'keras' 
#    
#    Network input should have structure of
#    
#    {"Input": tuple for convolution input (width,height,channels), int for data,
#    "HiddenLayers": List of hiddenlayers see below for structure, avalible options are:
#        Convolution ->
#        
#         {"Type": 'Conv2D',
#            "Width": Layer Width, int E.G 10
#            "Activation": Activation type E.G "relu","tanh","sigmoid",
#            "Kernel": Kernal Size, tuple E.G (3,3)
#            }
#        Dense ->
#        
#         {"Type": 'Dense',
#            "Width": Layer Width, int E.G 10,
#            "Activation":  Activation type E.G "relu","tanh","sigmoid""
#            }
#         
#        Dropout ->
#            {"Type": 'Dropout,
#            "Ratio": Ratio of neurons to drop out, float E.G 0.7
#            }
#        
#        Pooling -> 
#        {"Type": 'Pooling',
#        "kernal": Kernal Size, tuple E.G (3,3)}
#        "dim": dimensions of dropout, int E.G 1,2,3
#
#        } 
#   
#       Flatten - >
#        {"Type": 'Flatten,
#        } 
#    }    
#
#    """
#CANT HAVE FLATTERN AS FIRST LAYER YET    
    
    def __init__(self,Network,Backend,dataSlice = None):
        
        self.backend = Backend
        self.history = {"loss":[],"val_loss":[]}
        self.dataSlice = None

        #three types of model initilisation, from custom dictionary object, Empty Model, Direct Model Input
        if type(Network) == dict:
            self.input = Network["Input"]
            self.hiddenLayers = Network["HiddenLayers"]
            self.model = self.buildModel()
        elif (Network) == None:
            print("Empty Network Created,use model.loadModel(path,custom_object) function to load model.")
        else:# type(Network) == type(keras.engine.training.Model):
            self.model = Network

        if dataSlice!=None:
            self.loadData(dataSlice)
           
    def loadData(self,dataSliceObj):
        self.dataSlice = dataSliceObj  
    
    def buildModel(self):
        if self.backend == 'keras':
            return self.kerasModelBackend()
 
    def loadModel(self,name,customObject):
        #Get Current Working directory
        dir_path = os.getcwd()+"\Model_Saves\\"+name
        print(dir_path)
        if self.backend == 'keras':
            if customObject is not None:
                self.model = keras.models.load_model(dir_path+".h5",custom_objects=customObject)
            else:
                self.model = keras.models.load_model(dir_path+".h5")                
        
        with open(dir_path+ '.pkl', 'rb') as f:
            self.history = pickle.load(f)
        
    def saveModel(self,name):
        #Get Current Working directory
        dir_path = os.getcwd()+r"\Model_Saves/"
        #Check if directory exisits, create if not
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        #Save model in directory
        if self.backend == 'keras':
            self.model.save(dir_path+name+".h5")
        #Save history
        
        with open(dir_path+name+ '.pkl', 'wb') as f:
            pickle.dump(self.history, f, pickle.HIGHEST_PROTOCOL)
        
    def compileModel(self,Optimizer=None, Loss=None, Metrics= None ,Model=None):
        if self.backend== 'keras':
            self.kerasCompileModel(Optimizer,Loss,Metrics)            
        #Tensor flow just reqires Model
        if self.backend=='tensorflow':
            self.tfCompileModel(Model)

        
    def trainModel(self,Epochs = 100,Batch_size =None, Verbose = 2):

        if self.dataSlice != None:
            #If batch size is none set batch size ot size of training dataset
            if Batch_size is None:
                Batch_size = self.dataSlice.X_train.shape[0]
            # Training should return dictionary of loss ['loss'] and cross validation loss ['val_loss'] 
            if self.backend== 'keras':
                loss, val_loss = self.kerasTrainModel(Epochs,Batch_size,Verbose)
                print(self.history['loss'],loss)
                self.history['loss']+= loss
                
                if val_loss != None:
                    self.history['val_loss']+= val_loss
                else: 
                    self.history['val_loss'] = None
                
            elif self.backend == 'tensorflow':
                
                self.tfTrainModel(Epochs,Batch_size,Verbose)

        else:
            print("Please load data into model first using model.loadData(dataSlice)")
        
    def trainRoutine(self,routineSettings,trainRoutine):
        """            
               routineSettings = {
                "CompileAll":True,
                "SaveAll":"model.h5" or None
                }

            trainRoutine = [{
                "Compile":[Optimizer,Loss,Metrics],
                "Train":[Epochs,Batch_Size,Verbose]
                }]
        
        """
        if len(trainRoutine) == 0 or trainRoutine==None:
            print("Input Valid Train Routine")
            return 
            
        compileEach = routineSettings["CompileAll"]
        saveAll = routineSettings["SaveAll"]

        initCompile = trainRoutine[0]['Compile']
        self.compileModel(Optimizer=initCompile[0],Loss=initCompile[1],Metrics=initCompile[2])

        for routine in trainRoutine:

            if compileEach and routine != trainRoutine[0]:
                compSetting = routine['Compile']
                self.compileModel(Optimizer=compSetting[0],Loss=compSetting[1],Metrics=compSetting[2])
            trainSetting = routine['Train']
            self.trainModel(trainSetting[0],trainSetting[1],trainSetting[2])

            if saveAll != None:
                self.saveModel(saveAll)
        
        
    def generativeDataTrain(self,dataGenFunc, BatchSize=1, Epochs=100):

        for epoch in range(1,Epochs):
            data = []
            for i in range(0,BatchSize):
                item = dataGenFunc()
                data.append(np.array(item))
            
            data = np.array(data)
            print(data.shape)
            feat , labels,shape = self.channelOrderingFormat( np.array(data[0][0]), np.array(data[0][1]),256,256)
            print(feat.shape,labels.shape)
            loss =self.model.train_on_batch(feat,labels)
            print("loss",loss)

    def predictModel(self,testData):
        if self.backend== 'keras':
            return self.kerasPrecictModel(testData)

    def summary(self):
        if self.backend== 'keras':
            return self.model.summary()
    
    def gpuCheck(self):
        if self.backend== 'keras':
            keras.backend.tensorflow_backend._get_available_gpus()
    
    def getHistory(self):
        return self.history

    def clearHistory(self):
        self.history = []

    def plotLearningCurve(self,Loss_Val_Label="Validation Data",Loss_label="Training Data"):

        loss = []
        val_loss = []

        if self.backend== 'keras':
            loss,val_loss = self.kerasGetHistory()

        epochs = np.arange(0,len(loss),1)

        plt.plot(epochs,loss,label=Loss_label)

        if val_loss != None:
            plt.plot(epochs,val_loss,label=Loss_Val_Label)

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()


##################################       
        
### Keras Backend ############

##################################
    def channelOrderingFormat(self,Feat_train,Feat_test,img_rows,img_cols):
        if keras.backend.image_data_format() == 'channels_first':
            Feat_train = Feat_train.reshape(Feat_train.shape[0], 1, img_rows, img_cols)
            Feat_test = Feat_test.reshape(Feat_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            Feat_train = Feat_train.reshape(Feat_train.shape[0], img_rows, img_cols, 1)
            Feat_test = Feat_test.reshape(Feat_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)  
        return Feat_train, Feat_test, input_shape
    
    def kerasModelBackend(self):
            layers = []
            #Input layer
            print('num',len(self.hiddenLayers))

            dataModifier = False

            for layer in range(0,len(self.hiddenLayers)):    
                print('type',self.hiddenLayers[layer]["Type"])
                #Check for first layer to deal with input shape
                
                if layer == 0 or dataModifier:
                    #Convolution first layer
                    if self.hiddenLayers[layer]["Type"] == "Conv2D":
                        layers.append(keras.layers.Conv2D(self.hiddenLayers[layer]["Width"], self.hiddenLayers[layer]["Kernel"], input_shape=(self.input),padding="same",data_format= keras.backend.image_data_format(),activation=self.hiddenLayers[layer]["Activation"]))
                        dataModifier = False
                    #Dense first layer
                    elif self.hiddenLayers[layer]["Type"] == "Dense":
                        layers.append(keras.layers.Dense(self.hiddenLayers[layer]["Width"],input_dim=(self.input),kernel_initializer='normal', activation=self.hiddenLayers[layer]["Activation"]))
                        dataModifier = False
                    elif self.hiddenLayers[layer]["Type"] == "Flatten":
                        layers.append(keras.layers.Flatten())
                        dataModifier = True

                else:
                # 0 = convo2D 1 = dense 2 =Dropout, 3 = pooling, 4 = flatten 
                    if self.hiddenLayers[layer]["Type"] == "Conv2D":
                        layers.append(keras.layers.Conv2D(self.hiddenLayers[layer]["Width"], self.hiddenLayers[layer]["Kernel"], padding="same",data_format= keras.backend.image_data_format(),activation=self.hiddenLayers[layer]["Activation"]))
                    elif self.hiddenLayers[layer]["Type"] == "Dense":
                        layers.append(keras.layers.Dense(self.hiddenLayers[layer]["Width"],kernel_initializer='normal', activation=self.hiddenLayers[layer]["Activation"]))       
                    elif self.hiddenLayers[layer]["Type"] == "Dropout":
                        layers.append(keras.layers.Dropout(self.hiddenLayers[layer]["Ratio"]))    
                    elif self.hiddenLayers[layer]["Type"] == "Pool":
                        layers.append(keras.layers.MaxPooling2D(pool_size=self.hiddenLayers[layer]["Kernal"],data_format= keras.backend.image_data_format()))
                    elif self.hiddenLayers[layer]["Type"] == "Flatten":
                        layers.append(keras.layers.Flatten())
            return keras.Sequential(layers)               


    def kerasCompileModel(self,Optimizer,Loss,Metrics):
        self.model.compile(optimizer=Optimizer, loss=Loss, metrics=Metrics)
        
    def kerasTrainModel(self,Epochs,BatchSize,Verbose):
        history = self.model.fit(self.dataSlice.X_train, self.dataSlice.y_train, validation_data=(self.dataSlice.X_test,self.dataSlice.y_test), batch_size=BatchSize,epochs=Epochs, verbose=Verbose)
        
        if 'val_loss' in history.history:
            return history.history['loss'], history.history['val_loss']
        else:
            return history.history['loss'], None

    def kerasPrecictModel(self,testData):
        return self.model.predict(testData)
           
    def kerasGetHistory(self):
        return self.history['loss'],self.history['val_loss']


    
    def contourPlot(self):
        
        x1_min_tr = np.amin(self.dataSlice.X_train[:,0])
        x1_max_tr = np.amax(self.dataSlice.X_train[:,0])
        x2_min_tr = np.amin(self.dataSlice.X_train[:,1])
        x2_max_tr = np.amax(self.dataSlice.X_train[:,1])  

        x1_min_te = np.amin(self.dataSlice.X_test[:,0])
        x1_max_te = np.amax(self.dataSlice.X_test[:,0])
        x2_min_te = np.amin(self.dataSlice.X_test[:,1])
        x2_max_te = np.amax(self.dataSlice.X_test[:,1]) 
        
        x1_min = 0
        x1_max = 0
        x2_min = 0
        x2_max = 0
        
        if x1_min_tr > x1_min_te:
            x1_min = x1_min_te
        else:
            x1_min = x1_min_tr

        print(x2_min,x2_min_tr,x2_min_te)
        if x1_max_tr > x1_max_te:
            x1_max = x1_max_tr
        else:
            x1_max = x1_max_te


        if x2_min_tr > x2_min_te:
            x2_min = x2_min_te
        else:
            x2_min = x2_min_tr
        print(x2_min,x2_min_tr,x2_min_te)


        if x2_max_tr > x2_max_te:
            x2_max = x1_max_tr
        else:
            x2_max = x2_max_te
            
        xx, yy = np.meshgrid(np.arange(x1_min,x1_max,0.01),np.arange(x2_min,x2_max,0.01))            

        z = self.predictModel(np.c_[xx.ravel(),yy.ravel()])
        z = z.reshape(xx.shape)
        
        plt.contour(xx,yy,z)
        plt.scatter(self.dataSlice.X_train[:,0],self.dataSlice.X_train[:,1],c=self.dataSlice.y_train)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.colorbar()
        plt.show()



###################################      
        
### TensorFlow Backend ############

###################################
        
    def tfCompileModel(self,Model):
        #tf.InteractiveSession.close()
        self.Session = tf.InteractiveSession()
        self.Optimizer,self.Loss,self.Correct_Prediction, self.Metrics ,self.x,self.y= Model()
        
    def tfTrainModel(self,Epochs,BatchSize,Verbose):
        
        init_op = tf.global_variables_initializer()
#        with self.Session as sess:
            # initialise the variables
        self.Session.run(init_op)
        total_batch = int(self.dataSlice.X_train.shape[0]/BatchSize)
        for epoch in range(Epochs):
            avg_cost = 0
            for i in range(total_batch):
                
                batch_x, batch_y = self.dataSlice.getRandomBatch(BatchSize)
                
                _, c = self.Session.run([self.Optimizer, self.Loss], 
                                feed_dict={self.x: batch_x, self.y: batch_y})
                avg_cost += c / total_batch
            test_acc = self.Session.run(self.Metrics, 
                           feed_dict={self.x: self.dataSlice.X_test, self.y: self.dataSlice.y_test})
            print("Epoch:", str(epoch + 1), "cost =", "{:.3f}".format(avg_cost), "test accuracy: ","{:.3f}".format(test_acc))
    
        print("\nTraining complete!")
        print(self.Session.run(self.Metrics, feed_dict={self.x: self.dataSlice.X_test, self.y: self.dataSlice.y_test}))



        self.Session.close()






















#################################################################

## Pytorch Backend

###############################################################
"""
    def pyTorchModelBackend(self):
            layers = []
            activaions  = {"relu":torch.nn.Relu(True),"tanh":torch.nn.Tanh(True),"sigmoid":torch.nn.Sigmoid(True)}
            #Input layer
            print('num',len(self.hiddenLayers))
            for layer in range(0,len(self.hiddenLayers)):    
                print('type',self.hiddenLayers[layer]["Type"])
                #Check for first layer to deal with input shape
                
                if layer == 0:
                    #Convolution first layer
                    if self.hiddenLayers[layer]["Type"] == "Conv2D":
                        layers.append(torch.nn.Conv2D(self.input[2], self.hiddenLayers[layer]["Width"], kernal_size=(self.hiddenLayers["Kernal"])))
                        layers.append(activations[self.hiddenLayers[layer]["Activation"]])

                    #Dense first layer
                    elif self.hiddenLayers[layer]["Type"] == "Dense":
                        layers.append(torch.nn.Linear(self.input,self.hiddenLayers[layer]["Width"]))
                        layers.append(activations[self.hiddenLayers[layer]["Activation"]])
                else:
                # 0 = convo2D 1 = dense 2 =Dropout, 3 = pooling, 4 = flatten 
                    if self.hiddenLayers[layer]["Type"] == "Conv2D":
                        layers.append(torch.nn.Conv2D(self.input[2], self.hiddenLayers[layer]["Width"], kernal_size=(self.hiddenLayers["Kernal"])))
                        layers.append(activations[self.hiddenLayers[layer]["Activation"]])
                    elif self.hiddenLayers[layer]["Type"] == "Dense":
                        layers.append(torch.nn.Linear(self.input,self.hiddenLayers[layer]["Width"]))
                        layers.append(activations[self.hiddenLayers[layer]["Activation"]])
                    elif self.hiddenLayers[layer]["Type"] == "Dropout":
                        
                        if self.hiddenLayers[layer]["Dim"] == 1:
                            layers.append(torch.nn.Dropout(p=self.hiddenLayers[layer]["Ratio"]))
                        elif self.hiddenLayers[layer]["Dim"] == 2:
                            layers.append(torch.nn.Dropout2d(p=self.hiddenLayers[layer]["Ratio"]))
                        elif self.hiddenLayers[layer]["Dim"] == 3:
                            layers.append(torch.nn.Dropout3d(p=self.hiddenLayers[layer]["Ratio"]))
                    
                    elif self.hiddenLayers[layer]["Type"] == "Pool":
                        if self.hiddenLayers[layer]["Dim"] == 1:
                            layers.append(torch.nn.MaxPool1d(self.hiddenLayers[layer]["Kernal"]))
                        elif self.hiddenLayers[layer]["Dim"] == 1:
                            layers.append(torch.nn.MaxPool2d(self.hiddenLayers[layer]["Kernal"]))
                        elif self.hiddenLayers[layer]["Dim"] == 1:
                            layers.append(torch.nn.MaxPool3d(self.hiddenLayers[layer]["Kernal"]))
                    
                    elif self.hiddenLayers[layer]["Type"] == "Flatten":
                        layers.append(keras.layers.Flatten())
            return torch.nn.Sequential(*layers)
"""
