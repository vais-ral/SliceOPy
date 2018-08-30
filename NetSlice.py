import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from .DataSlice import DataSlice
import pickle

class NetSlice:

    def __init__(self,Network,Backend, Data_Slice = None,History_Keys = ["loss","val_loss"]):
        
        #Checking backend type
        if Backend not in ('keras'):
            sys.stderr.write("Please Enter Valid Backend, Suported Options are: 'keras'\n")
            sys.exit(0)
        else:
            os.environ['SLICE_BACKEND'] = "keras"
            global S
            from . import backends as S
        
        self.historyKeys = History_Keys
        #Set Empty History dictionary
        self.history = {}
        for item in self.historyKeys:
            self.history[item] = []
            
        
        #three types of model initilisation, from custom dictionary object, Empty Model, Direct Model Input
        if type(Network) == dict:
            self.input = Network["Input"]
            self.hiddenLayers = Network["HiddenLayers"]
            self.model = self.buildModel()
        elif (Network) == None:
            sys.stderr.write("Empty Network Model. Use Manual Model Loading model.loadModel(path,custom_object)\n")
        else:# type(Network) == type(keras.engine.training.Model):
            self.model = S.userBuildModel(Network)

        #DataSlice type checking
        if type(Data_Slice)== DataSlice:
            self.loadData(Data_Slice)
        elif Data_Slice==None:
            sys.stderr.write("Empty DataSlice Field, Please use manual data loading (.loadData())\n")
            self.dataSlice = None
        else:
            sys.stderr.write("Invalid Data Type Used, Input DataSlice Type Object\n")
            sys.exit(0)
           
    def loadData(self,dataSliceObj):
        if type(dataSliceObj) != DataSlice:
            sys.stderr.write("Invalid Data Type Used, Input DataSlice Type Object\n")
            sys.exit(0)
        else:
            self.dataSlice = dataSliceObj  
    
    def buildModel(self):
            return S.ModelBackend(self,self.hiddenLayers)
 
    def loadModel(self,name,customObject):
        #Get Current Working directory
        dir_path = os.getcwd()+"/Model_Saves/"+name
        
        self.model = S.userLoadModel(dir_path,name,customObject)

        try:
            with open(dir_path+ '.pkl', 'rb') as f:
                self.history = pickle.load(f)
        
            #load historyKeys from history loaded
                self.historyKeys = list(self.history.keys())
        except FileNotFoundError:
        # doesn't existprint()
            print("No History")
        else:
            print("No History")



    def clearHistory(self):
        for item in self.historyKeys:
            self.history[item] = []

    def saveModel(self,name):
        #Get Current Working directory
        dir_path = os.getcwd()+"/Model_Saves/"
  
        #Check if directory exisits, create if not
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
  
        #Save model in directory
        S.userSaveModel(self.model,dir_path+name+".h5")

        #Save history 
        with open(dir_path+name+ '.pkl', 'wb') as f:
            pickle.dump(self.history, f, pickle.HIGHEST_PROTOCOL)
        
    def compileModel(self,Optimizer=None, Loss=None, Metrics= None ,Model=None):     
        S.userCompileModel(self.model,Optimizer,Loss,Metrics)            

        
    def trainModel(self,Epochs = 100,Batch_size =None, Verbose = 2):

        if self.dataSlice != None:
            #If batch size is none set batch size ot size of training dataset
            if Batch_size is None:
                Batch_size = self.dataSlice.X_train.shape[0]

            # Training should return dictionary of loss ['loss'] and cross validation loss ['val_loss'] 
            self.history = S.userTrainModel(self.model,self.dataSlice,Epochs,Batch_size,Verbose,self.historyKeys,self.history)
            
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
        
        
    def generativeDataTrain(self,dataGenFunc, BatchSize=1, Epochs=100, Channel_Ordering=None):

        for epoch in range(0,Epochs):
            feat = []
            labels = []
            
            for i in range(0,BatchSize):
                item = dataGenFunc()
                feat.append(item[0])
                labels.append(item[1])
                
            feat = np.array(feat)
            labels= np.array(labels)
            
            if Channel_Ordering != None:
                feat , labels,shape = S.channelOrderingFormat(feat, labels,Channel_Ordering[0],Channel_Ordering[1],Channel_Ordering[2],Channel_Ordering[3])
            loss = S.userTrainOnBatch(self.model,feat,labels)
            
            self.history['loss'].append(loss)
            print("Epochs:",epoch+1, " Loss:",loss)
        
    def generativeDataTesting(self,dataGenFunc, SampleNumber=100,Threshold=1.0e-4, Channel_Ordering=None):
#
        feat = []
        labels = []
        for i in range(0,SampleNumber):
            item = dataGenFunc()
            feat.append(item[0])
            labels.append(item[1])
            
        feat = np.array(feat)
        labels= np.arrray(labels)
        
        if Channel_Ordering != None:
            feat , labels,shape = S.channelOrderingFormat(feat, labels,Channel_Ordering[0],Channel_Ordering[1],Channel_Ordering[2],Channel_Ordering[3])
#
#        feat = np.array(feat).reshape(len(feat),256,256)
#        labels = np.array(labels).reshape(len(feat),256,256)
#        feat , labels, shape = S.channelOrderingFormat(feat,labels,256,256)            
        predicted = self.predictModel(feat)
        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.imshow(featOrig[0])
        ax = fig.add_subplot(122)
        ax.imshow(predicted[0].reshape(256,256))
        plt.show()
        self.segmentationAccuracy(predicted,labels,Threshold)
    
    
    def segmentationAccuracy(self, predicted, labels,threshold):
        
        counter = np.subtract(predicted,labels)
        counter = np.abs(counter) < threshold
        counter = counter.flatten()
        positive = np.sum(counter)
        print((positive/counter.shape[0]))
        
        
        
    def predictModel(self,testData):
        return S.userPrecictModel(self.model,testData)

    def summary(self):
        return S.userSummary(self.model)
    
    def gpuCheck(self):
        S.userGPUCheck()
    
    def getHistory(self):
        return self.history


    def plotLearningCurve(self,Axes,Plot_Dict=None):

        for key in Plot_Dict.keys():
            if key in self.history.keys():
                data = self.history[key]
                Axes.plot(np.arange(0,len(data),1),data,label=Plot_Dict[key])

        Axes.set_xlabel('Epoch')
        Axes.set_ylabel('Loss')
        Axes.legend()

        return Axes


    
    def contourPlot(self,Axes):


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

        if x1_max_tr > x1_max_te:
            x1_max = x1_max_tr
        else:
            x1_max = x1_max_te


        if x2_min_tr > x2_min_te:
            x2_min = x2_min_te
        else:
            x2_min = x2_min_tr


        if x2_max_tr > x2_max_te:
            x2_max = x1_max_tr
        else:
            x2_max = x2_max_te
            
        xx, yy = np.meshgrid(np.arange(x1_min,x1_max,0.01),np.arange(x2_min,x2_max,0.01))            

        z = self.predictModel(np.c_[xx.ravel(),yy.ravel()])
        z = z.reshape(xx.shape)
        
        Axes.contour(xx,yy,z)
        Axes.scatter(self.dataSlice.X_train[:,0],self.dataSlice.X_train[:,1],c=self.dataSlice.y_train)
        Axes.set_xlabel('x1')
        Axes.set_ylabel('x2')
        return Axes




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
