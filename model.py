
################################################################################################################
### Contributers :Ahmed A                                                                                    ###
### Date: Feb-03-2022                                                                                        ###
### Description:  ConvLSTM Model Architecture                                                                ###
###                                                                                                          ###
### Refrences:                                                                                               ###
###              1.  https://github.com/cryptonymous9/Augmented-ConvLSTM/blob/master/model.py                ###
###              2.  Default Convolutional LSTM implementation is based on: https://arxiv.org/abs/1506.04214 ###
###              3.  Reccurennt drop based on https://arxiv.org/pdf/1603.05118                               ###
###              4.  https://medium.com/neuronio/an-introduction-to-convlstm-55c9025563a7                    ###
################################################################################################################


import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import (Dropout, MaxPooling2D, Flatten, Dense)

class SNNConvLSTM:    
    """
    Initialize convLSTM Layer with SNN

       Parameters:
           noChannel: Number of channels in input, climate variable of interest (RGB - 3) and Gray Scale (1)
           heightProj: Height dimension of projection
           widthProj: Width dimension of projection
           timeStep: Number of LSTM layers stacked on each other "Number of single projections"
    """
    def __init__(self, noChannel=1, heightProj=40, widthProj=40, timeStep=7):
        self.noChannel = noChannel
        self.heightProj = heightProj
        self.widthProj = widthProj
        self.timeStep = timeStep
     
    """
    Super Resolution
    """
    def SR_block(self, model_in, srBlock=[128,64,1], srBlockSize=[9, 5, 3]):
        #Input Imgae into model
        model = keras.layers.Conv2D(filters = srBlock[0], kernel_size = srBlockSize[0], padding='same', activation='relu')(model_in)
        #conv2D Layers
        model = keras.layers.Conv2D(filters = srBlock[1], kernel_size = srBlockSize[1], padding='same', activation='relu')(model)
        model = keras.layers.Conv2D(filters = srBlock[2], kernel_size = srBlockSize[2], padding='same', activation='relu')(model)

        #Output of the Layer Added
        output = keras.layers.Add()([model_in, model])
        #Return the Layer
        return output
    """
    ConvLSTM
    """
    def model(self, convLSTMK=[32,16,16], convLSTMKSize=[9,5,3], srBlock=[128,64,1], srBlockSize=[9,5,3],  srBlockDepth=1):
        #Input Layer of image
        model_in = keras.layers.Input(shape = (self.timeStep, self.heightProj, self.widthProj, self.noChannel))
        #Assign the ConvLSTM2D Model
        model = keras.layers.ConvLSTM2D(filters = convLSTMK[0], kernel_size = convLSTMKSize[0], padding='same', return_sequences = True)(model_in)
        #Normalization of Model
        model = keras.layers.BatchNormalization()(model)
        #ConvLSTM2D Model as part
        model = b = keras.layers.ConvLSTM2D(filters = convLSTMK[1], kernel_size = convLSTMKSize[1], padding='same', return_sequences = True)(model)
        #Normalization of Model
        model = keras.layers.BatchNormalization()(model)
        #ConvLSTM2D Model as part
        model = b = keras.layers.ConvLSTM2D(filters = convLSTMK[2], kernel_size = convLSTMKSize[2], padding = 'same', return_sequences = False)(model)
        #SNN Model Assignment
        for i in range(srBlockDepth):
            b = self.SR_block(b)
        #Conv2D Model layer
        b = keras.layers.Conv2D(filters = convLSTMK[-1], kernel_size = convLSTMKSize[-1], padding = 'same', activation = 'relu')(b)
        #Add Layer to model
        model = keras.layers.Add()([model, b])
        #conv2D Layer in model
        model = keras.layers.Conv2D(filters = 1, kernel_size = convLSTMKSize[-2], padding='same')(model)



        #Pooling 2Dm Dropout Value and Flatten the model
        model = MaxPooling2D(pool_size=(1, 1))(model)

        model = Dropout(0.5)(model)

        model = Flatten()(model)
        model= Dense(10, activation='relu')(model)
        model= Dense(1, activation='linear')(model)
        
        #Return the Model
        return keras.models.Model(model_in, model)
