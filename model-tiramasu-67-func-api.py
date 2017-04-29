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



from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D


import cv2
import numpy as np
import json

K.set_image_dim_ordering('tf')

# weight_decay = 0.0001
from keras.regularizers import l2
 
class Tiramisu():



    def __init__(self):
        self.create()

    def DenseBlock(self, layers_count, filters, previous_layer, model_layers, level):
        model_layers[level] = {}
        for i in range(layers_count):
            model_layers[level]['b_norm'+str(i+1)] = BatchNormalization(mode=0, axis=3,
                                         gamma_regularizer=l2(0.0001),
                                         beta_regularizer=l2(0.0001))(previous_layer)
            model_layers[level]['act'+str(i+1)] = Activation('relu')(model_layers[level]['b_norm'+str(i+1)])
            model_layers[level]['conv'+str(i+1)] = Conv2D(filters,   kernel_size=(3, 3), padding='same',
                                        kernel_initializer="he_uniform",
                                        data_format='channels_last')(model_layers[level]['act'+str(i+1)])
            model_layers[level]['drop_out'+str(i+1)] = Dropout(0.2)(model_layers[level]['conv'+str(i+1)])
            previous_layer  = model_layers[level]['drop_out'+str(i+1)]
        # print(model_layers)
        return model_layers[level]['drop_out'+ str(layers_count)] # return last layer of this level 


    def TransitionDown(self, filters, previous_layer, model_layers, level):
        model_layers[level] = {}
        model_layers[level]['b_norm'] = BatchNormalization(mode=0, axis=3,
                                     gamma_regularizer=l2(0.0001),
                                     beta_regularizer=l2(0.0001))(previous_layer)
        model_layers[level]['act'] = Activation('relu')(model_layers[level]['b_norm'])
        model_layers[level]['conv'] = Conv2D(filters, kernel_size=(1, 1), padding='same',
                                  kernel_initializer="he_uniform")(model_layers[level]['act'])
        model_layers[level]['drop_out'] = Dropout(0.2)(model_layers[level]['conv'])
        model_layers[level]['max_pool'] = MaxPooling2D(pool_size=(2, 2),
                                strides=(2, 2),
                                data_format='channels_last')(model_layers[level]['drop_out'])
        return model_layers[level]['max_pool']

    def TransitionUp(self,filters,input_shape,output_shape, previous_layer, model_layers, level):
        model_layers[level] = {}
        model_layers[level]['conv'] = Conv2DTranspose(filters,  kernel_size=(3, 3), strides=(2, 2),
                                            padding='same',
                                            output_shape=output_shape,
                                            input_shape=input_shape,
                                            kernel_initializer="he_uniform",
                                            data_format='channels_last')(previous_layer)

        return model_layers[level]['conv']

    def create(self):
        inputs = Input((224,224,3))

        first_conv = Conv2D(48, kernel_size=(3, 3), padding='same', 
                                kernel_initializer="he_uniform",
                                kernel_regularizer = l2(0.0001),
                                data_format='channels_last')(inputs)
        # first 

        enc_model_layers = {}

        layer_1_down  = self.DenseBlock(5,108, first_conv, enc_model_layers, 'layer_1_down' ) # 5*12 = 60 + 48 = 108
        layer_1a_down  = self.TransitionDown(108, layer_1_down, enc_model_layers,  'layer_1a_down')
        
        layer_2_down = self.DenseBlock(5,168, layer_1a_down, enc_model_layers, 'layer_2_down' ) # 5*12 = 60 + 108 = 168
        layer_2a_down = self.TransitionDown(168, layer_2_down, enc_model_layers,  'layer_2a_down')
        
        layer_3_down  = self.DenseBlock(5,228, layer_2a_down, enc_model_layers, 'layer_3_down' ) # 5*12 = 60 + 168 = 228
        layer_3a_down  = self.TransitionDown(228, layer_3_down, enc_model_layers,  'layer_3a_down')
        
        layer_4_down  = self.DenseBlock(5,288, layer_3a_down, enc_model_layers, 'layer_4_down' )# 5*12 = 60 + 228 = 288
        layer_4a_down  = self.TransitionDown(288, layer_4_down, enc_model_layers,  'layer_4a_down')
        
        layer_5_down  = self.DenseBlock(5,348, layer_4a_down, enc_model_layers, 'layer_5_down' ) # 5*12 = 60 + 288 = 348
        layer_5a_down  = self.TransitionDown(348, layer_5_down, enc_model_layers,  'layer_5a_down')

        layer_bottleneck  = self.DenseBlock(15,408, layer_5a_down, enc_model_layers,  'layer_bottleneck') # m = 348 + 5*12 = 408


        layer_1_up  = self.TransitionUp(468, (468, 7, 7), (None, 468, 14, 14), layer_bottleneck, enc_model_layers, 'layer_1_up')  # m = 348 + 5x12 + 5x12 = 468.
        skip_up_down_1 = concatenate([layer_1_up, enc_model_layers['layer_5_down']['conv'+ str(5)]], axis=-1)
        layer_1a_up  = self.DenseBlock(5,468, skip_up_down_1, enc_model_layers, 'layer_1a_up' )

        layer_2_up  = self.TransitionUp(408, (408, 14, 14), (None, 408, 28, 28), layer_1a_up, enc_model_layers, 'layer_2_up') # m = 288 + 5x12 + 5x12 = 408
        skip_up_down_2 = concatenate([layer_2_up, enc_model_layers['layer_4_down']['conv'+ str(5)]], axis=-1)
        layer_2a_up  = self.DenseBlock(5,408, skip_up_down_2, enc_model_layers, 'layer_2a_up' )

        layer_3_up  = self.TransitionUp(348, (348, 28, 28), (None, 348, 56, 56), layer_2a_up, enc_model_layers, 'layer_3_up') # m = 228 + 5x12 + 5x12 = 348
        skip_up_down_3 = concatenate([layer_3_up, enc_model_layers['layer_3_down']['conv'+ str(5)]], axis=-1)
        layer_3a_up  = self.DenseBlock(5,348, skip_up_down_3, enc_model_layers, 'layer_3a_up' )

        layer_4_up  = self.TransitionUp(288, (288, 56, 56), (None, 288, 112, 112), layer_3a_up, enc_model_layers, 'layer_4_up') # m = 168 + 5x12 + 5x12 = 288
        skip_up_down_4 = concatenate([layer_4_up, enc_model_layers['layer_2_down']['conv'+ str(5)]], axis=-1)
        layer_4a_up  = self.DenseBlock(5,288, skip_up_down_4, enc_model_layers, 'layer_4a_up' )

        layer_5_up  = self.TransitionUp(228, (228, 112, 112), (None, 228, 224, 224), layer_4a_up, enc_model_layers, 'layer_5_up') # m = 108 + 5x12 + 5x12 = 228
        skip_up_down_5 = concatenate([layer_5_up, enc_model_layers['layer_1_down']['conv'+ str(5)]], axis=-1)
        layer_5a_up  = self.DenseBlock(5,228, skip_up_down_5, enc_model_layers, 'concatenate' )

        # last 
        last_conv = Conv2D(12, activation='linear',
                                kernel_size=(1,1), 
                                padding='same',
                                kernel_regularizer = l2(0.0001),
                                data_format='channels_last')(layer_5a_up)
            
        reshape = Reshape((12, 224 * 224))(last_conv)
        perm = Permute((2, 1))(reshape)
        act = Activation('softmax')(perm)
        
        model = Model(inputs=[inputs], outputs=[act])

        with open('tiramisu_fc_dense67_model_12_func.json', 'w') as outfile:
            outfile.write(json.dumps(json.loads(model.to_json()), indent=3))

Tiramisu()