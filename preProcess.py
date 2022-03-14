
#################################################################################################################################################
### Contributers :Ahmed A                                                                                                                     ###
### Date: Jan-28-2022, Lastt Edit: Mar-04-2022                                                                                                ###
### Description:  Pre-proccess script                                                                                                         ###
###                                                                                                                                           ###
### Refrences: "Note that knowledge of pre-proccessing snippets are taken from these public codes but no explicit copy and paste was done"    ###
###                1.  https://github.com/giserh/ConvLSTM-2/blob/master/generate_lstm_data.py                                                 ###
###                2.  https://github.com/cryptonymous9/Augmented-ConvLSTM/blob/master/preprocess_data.py                                     ###
###                3.  https://www.kaggle.com/theblackmamba31/low-resolution-images-to-high-resolution                                        ###
###                4.  https://neptune.ai/blog/image-processing-python                                                                        ###
###                5.  https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/ ###
###                6.  https://www.geeksforgeeks.org/spatial-resolution-down-sampling-and-up-sampling-in-image-processing/                    ###
###                7.https://stackoverflow.com/questions/27710245/is-there-an-analysis-speed-or-memory-usage-advantage-to-using-hdf5-for-large-arr
###                8.https://www.christopherlovell.co.uk/blog/2016/04/27/h5py-intro.html                                                      ###
#################################################################################################################################################





#Preface
########
import numpy as np
from netCDF4 import Dataset
from sklearn import feature_extraction
import shutil
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image as im
import h5py


#Set directories and initial parameters
#######################################

#Directories
directory='Train'
coarseData='Data/Coarse'
fineData='Data/Fine'
predictData='Data/Predict'
ncOut='Output/outputNC'


#Year index
startYear_Train=1981
endYear_Train=2020

startYear_Predict=1950
endYear_Predict=1960

#Image Size and Defined Patch Range (0-27,0-49) parameters explored through metadata in prelimenary EDA file
imgSize=[40,40]
patch=[0,41]
imgValue=255.0 * 255.0

#Flag to Save Images and store NC files
saveImgFlag=False
saveNCFlag=False


#Data-preproccessing for training settings
###########################################
def tansform(fileLowRes,fileHighRes,filePredict,ncOutput):
    """
     This function is the main pre-proccessing script for ConvLSTM model  it consists of
     1.Getting data masked
     2.Image enhancment e.g."Noise reduction" by max value of climate attribute e.g."Preciptation"
     3.Up sampling using Cubic interpolation to match the spatial scale of cooridnates in high resolution images
     4.Patch Extraction

     :param fileLowRes: Low resolution Data as NetCDF4 format Path
     :param fileHighRes: High resolution Data as NetCDF4 format Path
     :param ncOutput: Low to High resolution data pre-proccessed  as NetCDF4 format output path
     :return: returns Low and High reolution patches in h5py format

     """
    print("Start Transforming")

    # Load Low Resolutions NC File, Extract preciptation values, mask and get Grid attributes of long/lat points
    ncHigh = Dataset(fileHighRes, 'r')

    dataHigh = ncHigh.variables['precip'][:]
    dataHigh[dataHigh.mask] = 0.0

    lonHigh = ncHigh.variables['longitude'][:]
    latHigh = ncHigh.variables['latitude'][:]
    
    ##Extract Endpoints of longitude and lalttitude
    stLonHigh=lonHigh[0]
    endLonHigh=lonHigh[len(lonHigh)-1]

    stLatHigh=latHigh[0]
    endLatHigh=latHigh[len(latHigh)-1]

    ##print Informatics
    print(fileHighRes)
    print('Longitude start & End :',stLonHigh,endLonHigh)    
    print('Latitude  start & End :',stLatHigh,endLatHigh)

    #Load Low Resolutions NC File, Extract preciptation values, mask and get Grid attributes of long/lat points
    ncLow = Dataset(fileLowRes, 'r')

    dataLow = ncLow.variables['tp'][:]
    dataLow[dataLow.mask] = 0.0

    lonLow = ncLow.variables['longitude'][:]
    latLow = ncLow.variables['latitude'][:]    

    ##Extract Endpoints of longitude and lalttitude
    stLonLow=lonLow[0]
    endLonLow=lonLow[len(lonLow)-1]

    stLatLow=latLow[0]
    endLatLow=latLow[len(latLow)-1]

    ##print Informatics
    print(fileLowRes)
    print('Longitude start & End :',stLonLow,endLonLow)    
    print('Latitude  start & End :',stLatLow,endLatLow)

    #Load Prediction Coarse  NC File, Extract preciptation values, mask and get Grid attributes of long/lat points
    ncPred = Dataset(filePredict, 'r')

    dataPred = ncPred.variables['tp'][:]
    dataPred[dataPred.mask] = 0.0

    lonPred= dataPred.variables['longitude'][:]
    latPred = dataPred.variables['latitude'][:]

    ##Extract Endpoints of longitude and lalttitude
    stLonPred=lonPred[0]
    endLonPred=lonPred[len(lonPred)-1]

    stLatPred=latPred[0]
    endLatPred=latPred[len(latPred)-1]

    ## print Informatics
    print(filePredict)
    print('Longitude start & End :',stLonPred,endLonPred)
    print('Latitude  start & End :',stLatPred,endLatPred)


    # print Shapes
    print("low shape", dataLow.shape)
    print("high shape", dataHigh.shape)
    print("Predict shape", dataPred.shape)

    if saveImgFlag:
        #Enhance Image using the Max constant value "refer to reference 4 and 5"
        maxHigh=np.max(dataHigh)
        maxLow=np.max(dataLow)
        maxPred=np.max(dataPred)
        normHigh=dataHigh/ maxHigh
        normLow=dataLow/ maxLow
        normPred=dataPred/maxPred
        #Save Daywise Images
        for i in range(len(normLow)):
            im.fromarray(normLow[i] * imgValue).convert("L").rotate(90,expand=True).save(f'Output/Low/low_{i+1}.png')
            im.fromarray(normHigh[i] * imgValue).convert("L").rotate(90,expand=True).save(f'Output/High/high_{i+1}.png')


    #Transform low resolution images to high rsolution images
    resScale = []    
    desireSize = dataHigh.shape[1:]
    for i in tqdm(np.arange(dataLow.shape[0]), total=dataLow.shape[0], desc="Transforming"):
        lowDay = dataLow[i, 4:225, 16:409]
        lowDay=cv2.flip(lowDay,0)
        resizeLow = cv2.resize(lowDay, (desireSize[1],desireSize[0]), interpolation=cv2.INTER_CUBIC)
        resScale.append(resizeLow.reshape([1, *desireSize]))

    resizeLow = np.concatenate(resScale, axis=0)
    print(resizeLow.shape)
    print(dataHigh.shape)

    # Transform low resolution images to high rsolution images
    resScalePredict = []
    desireSizePredict = dataHigh.shape[1:]
    for i in tqdm(np.arange(dataPred.shape[0]), total=dataPred.shape[0], desc="Transforming"):
        predictDay = dataLow[i, 4:225, 16:409]
        predictDay = cv2.flip(predictDay, 0)
        resizePredict = cv2.resize(predictDay, (desireSizePredict[1], desireSizePredict[0]),
                                   interpolation=cv2.INTER_CUBIC)
        resScalePredict.append(resizePredict.reshape([1, *desireSizePredict]))

    resizePredict = np.concatenate(resScalePredict, axis=0)
    print(resizePredict.shape)
    print(dataHigh.shape)

    #Save Daywise Images and NC file
    if saveImgFlag:
        maxLow=np.max(resizeLow)
        normLow=resizeLow/maxLow
        for i in range(len(normLow)):
            im.fromarray(normLow[i] * imgValue).convert("L").rotate(90,expand=True).save(f'Output/LowScale/low_scale_{i+1}.png')

    if saveNCFlag:
        shutil.copyfile(fileHighRes,ncOutput)
        ncOut = Dataset(ncOutput, 'r+')
        ncOut.variables['precip'][:]=resizePredict.astype(np.float16)
        ncOut.close()

    #Write Predict files
    shutil.copyfile(fileHighRes, ncOutput)
    ncOut = Dataset(ncOutput, 'r+')
    ncOut.variables['precip'][:] = resizeLow.astype(np.float16)
    ncOut.close()

    print("Complete Transforming")
    lowPatch=resizeLow[:,patch[0]*imgSize[0]:(patch[0]*imgSize[0])+imgSize[0],patch[1]*imgSize[1]:(patch[1]*imgSize[1])+imgSize[1]]
    highPatch=dataHigh[:,patch[0]*imgSize[0]:(patch[0]*imgSize[0])+imgSize[0],patch[1]*imgSize[1]:(patch[1]*imgSize[1])+imgSize[1]]
    print("Patch Extracted")
    return lowPatch, highPatch 


def saveFile(filenm,dataValues):
    """
        1.It is proven for time complexity that H5PY object proccess faster than normal arrays


        :param filenm: file name to be saved to
        :param datavalyes:Array patches from trnasform function
        :return: Nothing, Saves file in directory
        """
    h5f = h5py.File(filenm, 'w')
    h5f.create_dataset('values', data=dataValues)
    h5f.close()

if __name__ == "__main__":

    #Training part
    endYear_Train=endYear_Train+1

    ##Fine and Coarse Dataset in addition to output NC files
    fineList = [f"{fineData}/{directory}/Fine_{year}-01-01.nc" for year in range(startYear_Train, endYear_Train)]
    ##coarse Dataset
    coarseList = [f"{coarseData}/{directory}/Coarse_{year}-01-01.nc" for year in range(startYear_Train, endYear_Train)]
    ncOutList = [f"{ncOut}/low2High_{year}.nc" for year in range(startYear_Train, endYear_Train)]

    ##Initialize Patchs
    lowPatches=None
    highPatches=None

   ##Extract patches and apply pre-feined functions then concatnate patches
    for i in range(len(fineList)):
        lowPatch, highPatch=tansform(coarseList[i], fineList[i] ,ncOutList[i])
        try:
            lowPatches=np.concatenate((lowPatches, lowPatch))
            highPatches=np.concatenate((highPatches, highPatch))  
        except:
            lowPatches=lowPatch
            highPatches=highPatch

    ##Extract patches as H5PY format
    saveFile(f'Low_{startYear_Train}_{endYear_Train-1}.h5',lowPatches)
    saveFile(f'High_{startYear_Train}_{endYear_Train-1}.h5',highPatches)

    #Prediction file
    endYear_Predict = endYear_Predict + 1
    ##predict Dataset
    predictList = [f"{coarseData}/Coarse_{year}-01-01.nc" for year in range(startYear_Predict, endYear_Predict)]
    ##NC Output Files
    ncOutList = [f"{ncOut}/low2High_{year}.nc" for year in range(startYear_Predict, endYear_Predict)]

    ##for every files in directory, Normalization and Modify Image Scale & Cropping of Dataset
    for i in range(len(coarseList)):
        tansform(coarseList[i], fineData, ncOutList[i])





    
    

