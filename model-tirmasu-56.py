from __future__ import absolute_import
from __future__ import print_function
import os


import keras.models as models
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, Cropping2D
from keras.layers.normalization import BatchNormalization

from keras.layers import Conv2D, Conv2DTranspose

from keras import backend as K

import cv2
import numpy as np
import json

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
            model.add(Conv2D(filters, kernel_size=(3, 3), padding='same',
                                     kernel_initializer="he_uniform"))
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

    def TransitionUp(self,filters, input_shape,output_shape):
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

        self.DenseBlock(4,96) # 4*12 = 48 + 48 = 96
        self.TransitionDown(96)
        self.DenseBlock(4,144) # 4*12 = 48 + 96 = 144
        self.TransitionDown(144)
        self.DenseBlock(4,192) # 4*12 = 48 + 144 = 192
        self.TransitionDown(192)
        self.DenseBlock(4,240)# 4*12 = 48 + 192 = 240
        self.TransitionDown(240)
        self.DenseBlock(4,288) # 4*12 = 48 + 288 = 336
        self.TransitionDown(288)

        self.DenseBlock(15,336) # 4 * 12 = 48 + 288 = 336

        self.TransitionUp(384, (384, 7, 7), (None, 384, 14, 14))  # m = 288 + 4x12 + 4x12 = 384.
        self.DenseBlock(4,384)

        self.TransitionUp(336, (336, 14, 14), (None, 336, 28, 28)) #m = 240 + 4x12 + 4x12 = 336
        self.DenseBlock(4,336)

        self.TransitionUp(288, (288, 28, 28), (None, 288, 56, 56)) # m = 192 + 4x12 + 4x12 = 288
        self.DenseBlock(4,288)

        self.TransitionUp(240, (240, 56, 56), (None, 240, 112, 112)) # m = 144 + 4x12 + 4x12 = 240
        self.DenseBlock(4,240)

        self.TransitionUp(192, (192, 112, 112), (None, 192, 224, 224)) # m = 96 + 4x12 + 4x12 = 192
        self.DenseBlock(4,192)

        model.add(Conv2D(12, kernel_size=(1,1), 
                             padding='same',
                             kernel_initializer="he_uniform",
                             kernel_regularizer = l2(0.0001),
                            data_format='channels_last'))
        model.add(Reshape((12, 224 * 224)))
        model.add(Permute((2, 1)))
        model.add(Activation('softmax'))
        model.summary()

        with open('tiramisu_fc_dense56_model.json', 'w') as outfile:
            outfile.write(json.dumps(json.loads(model.to_json()), indent=3))

Tiramisu()