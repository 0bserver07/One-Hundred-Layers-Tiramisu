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

K.set_image_dim_ordering('th')

class Tiramisu():



    def __init__(self):
        self.create()

    def DenseBlock(self, layers, filters):
        model = self.model
        for i in range(layers):
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Conv2D(filters, kernel_size=(3, 3), padding='same'))
            model.add(Dropout(0.2))

    def TransitionDown(self,filters):
        model = self.model
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(filters, kernel_size=(1, 1), padding='same'))
        model.add(Dropout(0.2))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    def TransitionUp(self,filters, input_shape,output_shape):
        model = self.model
        model.add(Conv2DTranspose(filters,kernel_size=(3, 3), strides=(2, 2),data_format='channels_first', output_shape=output_shape,
                                    padding='same', input_shape=input_shape))


    def create(self):
        model = self.model = models.Sequential()
        # cropping
        # model.add(Cropping2D(cropping=((68, 68), (128, 128)), input_shape=(3, 360,480)))

        model.add(Conv2D(48, kernel_size=(3, 3), padding='same', input_shape=(3,224,224)))

        self.DenseBlock(4,112) # 4*16 = 64 + 48 = 112
        self.TransitionDown(112)
        self.DenseBlock(5,192) # 5*16 = 80 + 112 = 192
        self.TransitionDown(192)
        self.DenseBlock(7,304) # 7 * 16 = 112 + 192 = 304
        self.TransitionDown(304)
        self.DenseBlock(10,464)
        self.TransitionDown(464)
        self.DenseBlock(12,656)
        self.TransitionDown(656)

        self.DenseBlock(15,880)

        self.TransitionUp(1072, (7, 7,1072), (None, 14, 14, 1072))
        self.DenseBlock(12,1072)
        self.TransitionUp(800, (14, 14,800), (None, 28, 28, 800))
        self.DenseBlock(10,800)
        self.TransitionUp(560, (28, 28,560), (None, 56, 56, 560))
        self.DenseBlock(7,560)
        self.TransitionUp(368, (56, 56,368), (None, 112, 112, 368))
        self.DenseBlock(5,368)
        self.TransitionUp(256, (112, 112,256), (None, 224, 224, 256))
        self.DenseBlock(4,256)

        model.add(Conv2D(12, kernel_size=(3, 3), padding='same'))
        model.add(Reshape((12, 224 * 224)))
        model.add(Permute((2, 1)))
        model.add(Activation('softmax'))
        model.summary()

        with open('tiramisu_fc_dense103_model.json', 'w') as outfile:
            outfile.write(json.dumps(json.loads(model.to_json()), indent=3))

Tiramisu()