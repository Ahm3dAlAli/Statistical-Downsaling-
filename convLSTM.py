#########################################################################################################
### Contributers :Ahmed Ali                                                                             ###
### Date: Feb-06-2022                                                                                 ###
### Description:  ConvLSTM Model Training Script                                                      ###
###                                                                                                   ###
### Refrences:                                                                                        ###
###              1.https://www.analyticsvidhya.com/blog/2020/10/multivariate-multi-step-time-series-  ###
###                   forecasting-using-stacked-lstm-sequence-to-sequence-autoencoder-in-tensorflow-  ###
###                   2-0-keras/                                                                      ###
###              2. https://github.com/cryptonymous9/Augmented-ConvLSTM/blob/master/train.py          ###
#########################################################################################################




#Preface
##########

import argparse
import h5py
import os
from model import SNNConvLSTM
from sklearn.model_selection import train_test_split
import keras.preprocessing as prep
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras.losses import MeanSquaredError
from keras.losses import MeanAbsoluteError
from keras.optimizers import adam_v2
from tensorflow import image
from tensorflow import float32, constant_initializer
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np

#Setting path
#####################

#Training Dataset Path
lowTrain='Data/Low_1992_2020.h5'
highTrain='Data/High_1992_2020.h5'
#Testing Dataset Path
lowTest='Data/Low_1981_1987.h5'
highTest='Data/High_1981_1987.h5'
#Validate Dataset Path
lowValid='Data/Low_1988_1991.h5'
highValid='Data/High_1988_1991.h5'

#Training Parameters
#####################

#No. of Channel
noChannel=1
#Image Resolutions
dimProjection=[40,40]
#No. of Days as time stemp
timeStep=7
#Batch Size
batchSize=64
#Learning Rate
learningRate=0.0001
#Number of Epochs
noEpochs=10

# Reading files
###############

def readNCFile(ncFile,label='precip'):
    data=Dataset(ncFile, 'r')   
    return data.variables[label][:]

def readH5File(h5File,label='values'):
    data= h5py.File(h5File, 'r')
    dataV=np.array(data.get(label))
    return dataV

# Loss metric
##############

def PSNRLoss(orig, predict):
    orig=K.sqrt(orig)
    predict=K.sqrt(predict)
    img1 = image.convert_image_dtype(orig, float32)
    img2 = image.convert_image_dtype(predict, float32)
    return image.psnr(img2, img1, max_val=1.0)


# Normlaization
###############
def NormalizeData(data):
    Max=np.max(data)
    return data/Max, Max

# Data split settings as a time series
######################################
#Data Generation from Original Data
def dataset(trainX, trainY, testX, testY, validX, validY):

    train = prep.sequence.TimeseriesGenerator(trainX, trainY,length=timeStep, batch_size=batchSize)
    test = prep.sequence.TimeseriesGenerator(testX, testY,length=timeStep, batch_size=batchSize)
    valid = prep.sequence.TimeseriesGenerator(validX, validY,length=timeStep, batch_size=batchSize)

    return train, test, valid

if __name__ == "__main__":

    trainX=readH5File(lowTrain)
    trainY=readH5File(highTrain)


    testX=readH5File(lowTest)
    testY=readH5File(highTest)


    validX=readH5File(lowValid)
    validY=readH5File(highValid)


    trainX, mTrainX = NormalizeData(trainX)
    testX, mTestX = NormalizeData(testX)
    validX, mValidX = NormalizeData(validX)
    trainY, mTrainY = NormalizeData(trainY)
    testY, mTestY = NormalizeData(testY)
    validY, mValidY = NormalizeData(validY)    

    train, test, valid  = dataset(trainX, trainY, testX, testY, validX, validY)


    convLSTMSNN = SNNConvLSTM
    model = convLSTMSNN().model()
    print(model.summary())    

    #Save the Model for every check points
    path="output/model/"+"cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(path)

    #Saving Model Parameters
    check_model = ModelCheckpoint(filepath=path, verbose=1, save_weights_only=True,  monitor='loss', save_best_only=True)

    #Model Compile Parameters
    model.compile(loss=MeanSquaredError(), optimizer=adam_v2(learning_rate=learningRate), metrics=['mae',PSNRLoss])
  
    #Model Generation with Training and Testing Dataset     
    history=model.fit(train, steps_per_epoch=trainY.shape[0]//batchSize, epochs=noEpochs, validation_data=valid, validation_steps=validY.shape[0]//batchSize, callbacks=[check_model])

    #Evalute The Model
    score = model.evaluate(test, verbose=0)

    #Print Model Results on Testing Dataset
    print(model.metrics_names)
    print(score)

    #Save Model of Training
    model.save('Model/')

    #Plot the Graph of PSNRLoss V/s Epochs List of PSNR Loss of each Epoch ('mae','loss','PSNRLoss','val_mae','val_loss','val_PSNRLoss')
    loss_val = history.history['mae']
    plt.plot(history.epoch, loss_val, 'g', label='Mean Absolute Error')
    plt.title('Mean Absolute Error v/s Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('Output/plot.png')
    plt.show()
