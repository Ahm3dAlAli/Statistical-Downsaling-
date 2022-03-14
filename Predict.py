#########################################################################################################
### Contributers :Ahmed Ali                                                                           ###
### Date: Feb-24-2022                                                                                 ###
###                                                                                                   ###
#########################################################################################################




import keras.backend as K
from tensorflow import image
from tensorflow import float32
from netCDF4 import Dataset 
import numpy as np
import tensorflow as tf
from skimage.util import view_as_blocks
import shutil

#Setting path
#####################
#Pre-proccessed datan path
lowTest='Output/outputNC/low2High_1960.nc'
#NC File Output path
ncOutput='Output/predictNC/predict_1960.nc'

#Prediction Parameters
#####################
dimProjection=[40,40]


# Reading files
###############
def readNCFile(ncFile,label='precip'):
    data=Dataset(ncFile, 'r')   
    return data.variables[label][:]

# Loss Metric
###############
def PSNRLoss(orig, predict):
    orig=K.sqrt(orig)
    predict=K.sqrt(predict)
    img1 = image.convert_image_dtype(orig, float32)
    img2 = image.convert_image_dtype(predict, float32)
    return image.psnr(img2, img1, max_val=1.0)

# Normlaization
###############
def NoiseReduction(data):
    Max=np.max(data)
    return data/Max, Max     

if __name__ == "__main__":   
    
    #Read Testing Dataset
    testX=readNCFile(lowTest)

    #Add Padding
    testX=np.pad(testX, [(0, 0), (0, 20), (0, 0)], mode='constant')

    #Noise reduction of all Dataset
    testX, mTestX = NoiseReduction(testX)
    print(testX.shape)

    #Load Model    
    model = tf.keras.models.load_model('imageModel/', custom_objects={'PSNRLoss':PSNRLoss})    
    print(model.summary())

    for k in range(len(testX)):
        print(k)
        #Each Day Image Predict    
        B=view_as_blocks(testX[k], (dimProjection[0], dimProjection[1]))
        for i in range(len(B)):
            for j in range(len(B[0])):
                #Predict Image using pre-Model
                testXY=B[i][j]            
                testing=testXY.reshape(-1, 1, dimProjection[0], dimProjection[1]) 
                pred = model.predict(testing)
                B[i][j]=pred[0,:,:,0] 
        testX[k]=B.transpose(0,2,1,3).reshape(-1,B.shape[1]*B.shape[3])
        #A = gaussian_filter(A, sigma=7)


    #Copy File
    shutil.copyfile(lowTest,ncOutput)
    #Open copied File 
    ncOut = Dataset(ncOutput, 'r+')
    #Change Update Values
    ncOut.variables['precip'][:]=test.astype(np.float16)
    #Close File
    ncOut.close()
    

    

    
    
