#########################################################################################################
### Contributers :Ahmed Ali                                                                             ###
### Date: Feb-16-2022                                                                                 ###
###                                                                                                   ###
#########################################################################################################




import argparse
import h5py
import os
from keras.models import load_model
import keras.preprocessing as prep
import keras.backend as K
from tensorflow import image
from tensorflow import float32
from netCDF4 import Dataset 
import numpy as np
import tensorflow as tf



#Setting path
#####################
#Testing Dataset Path
lowTest='Data/Low_1981_1987.h5'
highTest='Data/High_1981_1987.h5'

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


# Reading files
###############
def readNCFile(ncFile,label='precip'):
    data=Dataset(ncFile, 'r')   
    return data.variables[label][:]

def readH5File(h5File,label='values'):
    data= h5py.File(h5File, 'r')
    dataV=np.array(data.get(label))
    lenV=int(len(dataV)/25)
    return dataV[0:lenV,:,:]

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

if __name__ == "__main__":   
    #Read Testing Dataset
    testX=readH5File(lowTest)
    testY=readH5File(highTest)
    
    #Normalization of all Dataset
    testX, mTestX = NormalizeData(testX)
    testY, mTestY = NormalizeData(testY)   

    #Testing Dataset
    test = prep.sequence.TimeseriesGenerator(testX, testY, length=timeStep, batch_size=batchSize)

    #Load Model    
    model = tf.keras.models.load_model('wholeModel/', custom_objects={'PSNRLoss':PSNRLoss})    
    print(model.summary())    
    
    #Evalute The Model
    score = model.evaluate(test)

    #Print Model Results on Testing Dataset
    print(model.metrics_names)
    print(score)  
    
