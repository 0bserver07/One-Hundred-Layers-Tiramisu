from __future__ import absolute_import
from __future__ import print_function
import os


import keras.models as models
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Merge, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, Deconvolution2D, Cropping2D
from keras.layers.normalization import BatchNormalization


from keras import backend as K

import cv2
import numpy as np
import json



class Tiramisu():



    def __init__(self):
        self.create()

    def DenseBlock(self, layers, filters):
        model = self.model
        for i in range(layers):
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Convolution2D(filters, 3, 3, border_mode='same'))
            model.add(Dropout(0.2))
    
    def TransitionDown(self,filters):
        model = self.model
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution2D(filters, 1, 1, border_mode='same'))
        model.add(Dropout(0.2))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    def TransitionUp(self,filters, input_shape,output_shape):
        model = self.model
        model.add(Deconvolution2D(filters, 3, 3, output_shape=output_shape, subsample=(2, 2),
                                    border_mode='same', input_shape=input_shape))


    def create(self):
        model = self.model = models.Sequential()
        # cropping
        # model.add(Cropping2D(cropping=((68, 68), (128, 128)), input_shape=(3, 360,480)))
        
        model.add(Convolution2D(48, 3, 3, border_mode='same', input_shape=(3,224,224)))
        
        self.DenseBlock(4,112)
        self.TransitionDown(112)
        self.DenseBlock(5,192)
        self.TransitionDown(192)
        self.DenseBlock(7,304)
        self.TransitionDown(304)
        self.DenseBlock(10,464)
        self.TransitionDown(464)
        self.DenseBlock(12,656)
        self.TransitionDown(656)

        self.DenseBlock(15,880)
        
        self.TransitionUp(1072, (1072, 7, 7), (None, 1072, 14, 14))
        self.DenseBlock(12,1072)
        self.TransitionUp(800, (800, 14, 14), (None, 800, 28, 28))
        self.DenseBlock(10,800)
        self.TransitionUp(560, (800, 28, 28), (None, 800, 56, 56))
        self.DenseBlock(7,560)
        self.TransitionUp(368, (800, 56, 56), (None, 800, 112, 112))
        self.DenseBlock(5,368)
        self.TransitionUp(256, (800, 112, 112), (None, 800, 224, 224))
        self.DenseBlock(4,256)
        
        model.add(Convolution2D(12, 1, 1, border_mode='same'))
        model.add(Reshape((12, 224 * 224)))
        model.add(Permute((2, 1)))
        model.add(Activation('softmax'))
        model.summary()
        
        with open('tiramisu_fc_dense103_model.json', 'w') as outfile:
            outfile.write(json.dumps(json.loads(model.to_json()), indent=3))

Tiramisu()