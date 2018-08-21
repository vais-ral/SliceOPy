# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 13:26:22 2018

@author: lhe39759
"""
import math
import numpy as np
import keras

class DataSlice:

    """ Initialise with two numpy arrays for features and labels"""


    def __init__(self,Features = None,Labels = None,Shuffle=True,Rebalance = 0.0, Split_Ratio = 0.7,Channel_Features = None, Channel_Labels = None,info=True):
        
        self.info = True
        self.Shuffle = Shuffle
        self.Rebalance = Rebalance
        self.Split_Ratio = Split_Ratio

        if Features is not None and Labels is not None:

            if info:
                print("-----------------------------DataSlice------------------------")
                print("Number of Features:", Features.shape[0], "Number of Labels:", Labels.shape[0])
                print("Feature Shape:", Features.shape[1:])
            
            if Shuffle:
                Features ,Labels = self.shuffleData(Features,Labels)
                if info:
                    print("Data Shuffled")

            self.X_train,self.X_test,self.y_train,self.y_test = self.splitData(Features,Labels,Split_Ratio)

            if Rebalance != None:
                self.X_train,self.y_train = self.reBalanceData(self.X_train,self.y_train,Rebalance)
                if info:    
                    print("Data Rebalnced, Ratio:",Rebalance)
            if info:  
                print("Training Data:",self.X_train.shape[0])
                print("Test Data:",self.X_test.shape[0])


            if Channel_Features != None:
                self.channelOrderingFormatFeatures(Channel_Features[0],Channel_Features[1])
                if info:
                    print("Channel Ordering Features, New Feature Shape:", self.X_train.shape[1:])
            if Channel_Labels != None:
                self.channelOrderingFormatLabels(Channel_Labels[0],Channel_Labels[1])
                if info:
                    print("Channel Ordering Label, New Label Shape:", self.X_train.shape[1:])
            if info:  
                print("-------------------------------------------------------------")

        else:
            print("Empty DataSlice Object Created. Use Manual DataSlice.loadFeatTraining(), DataSlice.loadFeatTest(), DataSlice.loadLabelTraining(), DataSlice.loadLabelTest() Methods ")


    def featureColumn(self,columns):
        self.X_test = self.X_test[:,columns]
        self.X_train = self.X_train[:,columns]


    def loadFeatTraining(self,data):
        self.X_train = data

    def loadFeatTest(self,data):
        self.X_test = data

    def loadLabelTraining(self,data):
        self.y_train = data

    def loadLabelTest(self,data):
        self.y_test = data

    def channelOrderingFormatFeatures(self,img_rows,img_cols):
        self.X_train,self.X_test,input_shape = self.channelOrderingFormat(self.X_train,self.X_test,img_rows,img_cols)
        return input_shape
        
    def channelOrderingFormatLabels(self,img_rows,img_cols):
        self.y_train, self.y_test,input_shape = self.channelOrderingFormat(self.y_train, self.y_test,img_rows,img_cols)
        return input_shape

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

    def reBalanceData(self,x,y,Multip):
        
            ones = x[np.where(y==1)].copy()
            y_ones = y[np.where(y==1)].copy()
            total = len(y)
            total_one = len(ones)
            multiplier = int(math.ceil((total/total_one)*Multip))
            for i in range(multiplier):
                x = np.insert(x,1,ones,axis=0)
                y = np.insert(y,1,y_ones,axis=0)
        
            ran = np.arange(x.shape[0])
            np.random.shuffle(ran)
            
            return x[ran], y[ran]
        

    def splitData(self,features,labels,ratio):
        length = features.shape[0]
        return features[:int(length*ratio)],features[int(length*ratio):],labels[:int(length*ratio)],labels[int(length*ratio):]

    def shuffleData(self,features,labels):
        
        ran = np.arange(features.shape[0])
        np.random.shuffle(ran)
        features= features[ran]
        labels= labels[ran]
        
        return features,labels

    def saveData(self,path):
        np.save(path+'_Shuffle_'+str(self.Shuffle)+'Rebalance_'+str(self.Rebalance)+'Split_Ratio_'+str(self.Split_Ratio)+'_X_train',self.X_train)
        np.save(path+'_Shuffle_'+str(self.Shuffle)+'Rebalance_'+str(self.Rebalance)+'Split_Ratio_'+str(self.Split_Ratio)+'_X_test',self.X_test)
        np.save(path+'_Shuffle_'+str(self.Shuffle)+'Rebalance_'+str(self.Rebalance)+'Split_Ratio_'+str(self.Split_Ratio)+'_y_train',self.y_train)
        np.save(path+'_Shuffle_'+str(self.Shuffle)+'Rebalance_'+str(self.Rebalance)+'Split_Ratio_'+str(self.Split_Ratio)+'_y_test',self.y_test)

    def oneHot(self,outSize):
        self.y_train = self.convertOneHot(self.y_train,outSize)
        self.y_test = self.convertOneHot(self.y_test,outSize)

    def convertOneHot(self,labels,out):
        label = np.zeros((labels.shape[0],out),dtype=np.float32)
        for i in range(0,len(labels)):
            if labels[i] == 0:
                label[i,:] = np.array([0,1])
            else:
                label[i,:] = np.array([1,0])
        return label
    

        
    def imagePadArray(self,image,segment):
        return np.array(np.pad(image,segment,'constant', constant_values=0))
    
    def getRandomBatch(self,batchSize):
        
        ran = np.arange(self.X_train.shape[0])
        np.random.shuffle(ran)
        
        return self.X_train[ran][:batchSize],self.y_train[ran][:batchSize]