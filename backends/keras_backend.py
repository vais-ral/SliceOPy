import keras 
import keras.backend as K

##################################       
    
### Keras Backend ############

##################################

def userBuildModel(Network):
    return Network

def userLoadModel(dir_path,name,customObject):
    
    if customObject is not None:
        print(dir_path)
        return keras.models.load_model(dir_path+".h5",custom_objects=customObject)
    else:
        return keras.models.load_model(dir_path+".h5")                
        
def userSaveModel(model,path):
    model.save(path)

def convertOneHot(labels,out):
    
    label = np.zeros((labels.shape[0],out),dtype=np.float32)
    for i in range(0,len(labels)):
        if labels[i] == 0:
            label[i,labels[i]] = 1
        else:
            label[i,labels[i]] = 1
    return label

def channelOrderingFormat(Feat_train,img_rows,img_cols,c1):
    if K.image_data_format() == 'channels_first':
        Feat_train = Feat_train.reshape(Feat_train.shape[0], c1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        Feat_train = Feat_train.reshape(Feat_train.shape[0], img_rows, img_cols, c1)
        input_shape = (img_rows, img_cols, 1)  
    return Feat_train, input_shape           

def userCompileModel(model,Optimizer,Loss,Metrics):
    model.compile(optimizer=Optimizer, loss=Loss, metrics=Metrics)
    
def userTrainModel(model,dataSlice,Epochs,BatchSize,Verbose,historyKeys,historySave):
    history = model.fit(dataSlice.X_train, dataSlice.y_train, validation_data=(dataSlice.X_test,dataSlice.y_test), batch_size=BatchSize,epochs=Epochs, verbose=Verbose)
    for item in historyKeys:
        if item in history.history.keys():
            historySave[item] =historySave[item]+history.history[item]
        else:
            sys.stderr.write("Key: "+ item+" does not exist in keras history \n")

    return historySave

def userTrainOnBatch(model,feat,labels):
    return model.train_on_batch(feat,labels)


def userPrecictModel(model,testData):
    return model.predict(testData)
        
def userSummary(model):
    return model.summary()

def userGPUCheck():
    keras.backend.tensorflow_backend._get_available_gpus()

def ModelBackend(self,hiddenLayers):
        layers = []
        #Input layer

        dataModifier = False

        for layer in range(0,len(hiddenLayers)):    
            #Check for first layer to deal with input shape
            
            if layer == 0 or dataModifier:
                #Convolution first layer
                if hiddenLayers[layer]["Type"] == "Conv2D":
                    layers.append(keras.layers.Conv2D(hiddenLayers[layer]["Width"], hiddenLayers[layer]["Kernel"], input_shape=(self.input),padding="same",data_format= keras.backend.image_data_format(),activation=hiddenLayers[layer]["Activation"]))
                    dataModifier = False
                #Dense first layer
                elif hiddenLayers[layer]["Type"] == "Dense":
                    layers.append(keras.layers.Dense(hiddenLayers[layer]["Width"],input_dim=(self.input),kernel_initializer='normal', activation=hiddenLayers[layer]["Activation"]))
                    dataModifier = False
                elif hiddenLayers[layer]["Type"] == "Flatten":
                    layers.append(keras.layers.Flatten())
                    dataModifier = True

            else:
            # 0 = convo2D 1 = dense 2 =Dropout, 3 = pooling, 4 = flatten 
                if hiddenLayers[layer]["Type"] == "Conv2D":
                    layers.append(keras.layers.Conv2D(hiddenLayers[layer]["Width"], hiddenLayers[layer]["Kernel"], padding="same",data_format= keras.backend.image_data_format(),activation=hiddenLayers[layer]["Activation"]))
                elif hiddenLayers[layer]["Type"] == "Dense":
                    layers.append(keras.layers.Dense(hiddenLayers[layer]["Width"],kernel_initializer='normal', activation=hiddenLayers[layer]["Activation"]))       
                elif hiddenLayers[layer]["Type"] == "Dropout":
                    layers.append(keras.layers.Dropout(hiddenLayers[layer]["Ratio"]))    
                elif hiddenLayers[layer]["Type"] == "Pool":
                    layers.append(keras.layers.MaxPooling2D(pool_size=hiddenLayers[layer]["Kernal"],data_format= keras.backend.image_data_format()))
                elif hiddenLayers[layer]["Type"] == "Flatten":
                    layers.append(keras.layers.Flatten())
        return keras.Sequential(layers)  