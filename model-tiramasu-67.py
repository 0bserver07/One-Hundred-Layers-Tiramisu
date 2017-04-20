from __future__ import absolute_import
from __future__ import print_function
import os


import keras.models as models
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, Cropping2D
from keras.layers.normalization import BatchNormalization

from keras.layers import add

from keras.layers import Conv2D, Conv2DTranspose

from keras import backend as K

import cv2
import numpy as np
import json

K.set_image_dim_ordering('tf')

# weight_decay = 0.0001
from keras.regularizers import l2
 
class Tiramisu():



    def __init__(self):
        self.create()

    def DenseBlock(self, layers, filters):
        model = self.model
        for i in range(layers):
            model.add(BatchNormalization(mode=0, axis=1,
                                         gamma_regularizer=l2(0.0001),
                                         beta_regularizer=l2(0.0001)))
            model.add(Activation('relu'))
            model.add(Conv2D(filters,   kernel_size=(3, 3), padding='same',
                                        kernel_initializer="he_uniform",
                                        data_format='channels_last'))
            model.add(Dropout(0.2))


    def TransitionDown(self,filters):
        model = self.model
        model.add(BatchNormalization(mode=0, axis=1,
                                     gamma_regularizer=l2(0.0001),
                                     beta_regularizer=l2(0.0001)))
        model.add(Activation('relu'))
        model.add(Conv2D(filters, kernel_size=(1, 1), padding='same',
                                  kernel_initializer="he_uniform"))
        model.add(Dropout(0.2))
        model.add(MaxPooling2D( pool_size=(2, 2),
                                strides=(2, 2),
                                data_format='channels_last'))

    def TransitionUp(self,filters,input_shape,output_shape):
        model = self.model
        model.add(Conv2DTranspose(filters,  kernel_size=(3, 3), strides=(2, 2),
                                            padding='same',
                                            output_shape=output_shape,
                                            input_shape=input_shape,
                                            kernel_initializer="he_uniform",
                                            data_format='channels_last'))


    def create(self):
        model = self.model = models.Sequential()
        # cropping
        # model.add(Cropping2D(cropping=((68, 68), (128, 128)), input_shape=(3, 360,480)))

        model.add(Conv2D(48, kernel_size=(3, 3), padding='same', 
                             input_shape=(224,224,3),
                            kernel_initializer="he_uniform",
                            kernel_regularizer = l2(0.0001),
                            data_format='channels_last'))
        # (5 * 4)* 2 + 5 + 5 + 1 + 1 +1
        # growth_m = 4 * 12
        # previous_m = 48
        self.DenseBlock(5,108) # 5*12 = 60 + 48 = 108
        self.TransitionDown(108)
        self.DenseBlock(5,168) # 5*12 = 60 + 108 = 168
        self.TransitionDown(168)
        self.DenseBlock(5,228) # 5*12 = 60 + 168 = 228
        self.TransitionDown(228)
        self.DenseBlock(5,288)# 5*12 = 60 + 228 = 288
        self.TransitionDown(288)
        self.DenseBlock(5,348) # 5*12 = 60 + 288 = 348
        self.TransitionDown(348)

        self.DenseBlock(15,408) # m = 348 + 5*12 = 408


        self.TransitionUp(468, (468, 7, 7), (None, 468, 14, 14))  # m = 348 + 5x12 + 5x12 = 468.
        self.DenseBlock(5,468)

        self.TransitionUp(408, (408, 14, 14), (None, 408, 28, 28)) # m = 288 + 5x12 + 5x12 = 408
        self.DenseBlock(5,408)

        self.TransitionUp(348, (348, 28, 28), (None, 348, 56, 56)) # m = 228 + 5x12 + 5x12 = 348
        self.DenseBlock(5,348)

        self.TransitionUp(288, (288, 56, 56), (None, 288, 112, 112)) # m = 168 + 5x12 + 5x12 = 288
        self.DenseBlock(5,288)

        self.TransitionUp(228, (228, 112, 112), (None, 228, 224, 224)) # m = 108 + 5x12 + 5x12 = 228
        self.DenseBlock(5,228)

        model.add(Conv2D(12, kernel_size=(1,1), 
                             padding='same',
                             kernel_initializer="he_uniform",
                             kernel_regularizer = l2(0.0001),
                            data_format='channels_last'))
        
        model.add(Reshape((12, 224 * 224)))
        model.add(Permute((2, 1)))
        model.add(Activation('softmax'))
        model.summary()

        with open('tiramisu_fc_dense67_model_12.json', 'w') as outfile:
            outfile.write(json.dumps(json.loads(model.to_json()), indent=3))

Tiramisu()