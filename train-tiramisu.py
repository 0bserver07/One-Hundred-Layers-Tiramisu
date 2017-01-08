from __future__ import absolute_import
from __future__ import print_function
import os

os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=gpu0,floatX=float32,optimizer=None'

# import sys;
# sys.setrecursionlimit(40000)




import keras.models as models
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Merge, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, Deconvolution2D, Cropping2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop
from keras import backend as K
from keras.layers import Input, merge
from keras.regularizers import l2
from keras.layers.pooling import AveragePooling2D
from keras.models import Model

import cv2
import numpy as np
import json
np.random.seed(07) # 0bserver07 for reproducibility




class_weighting = [
 0.2595,
 0.1826,
 4.5640,
 0.1417,
 0.5051,
 0.3826,
 9.6446,
 1.8418,
 6.6823,
 6.2478,
 3.0,
 7.3614
]


# 11 class
# [
#   0.2595,
#   0.1826,
#   4.5640,
#   0.1417,
#   0.9051,
#   0.3826,
#   9.6446,
#   1.8418,
#   0.6823,
#   6.2478,
#   7.3614,
# ]

# load the data
train_data = np.load('./data/train_data.npy')
train_label = np.load('./data/train_label.npy')

test_data = np.load('./data/test_data.npy')
test_label = np.load('./data/test_label.npy')

# load the model:
# with open('tiramisu_fc_dense103_model.json') as model_file:
#     tiramisu = models.model_from_json(model_file.read())


nb_layers = [4, 5, 7, 10, 12, 15]
tiramisu = create_fc_dense_net(nb_classes=12,img_dim=(3, 224, 224), nb_layers=nb_layers)

# section 4.1 from the paper
optimizer = RMSprop(lr=1e-03, rho=0.9, epsilon=1e-08, decay=0.995)
tiramisu.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# checkpoint
filepath="tiramisu_weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1,
									 save_best_only=True, mode='max')
callbacks_list = [checkpoint]

nb_epoch = 100
batch_size = 3

# 
tiramisu.load_weights('weights/tiramisu_model_weight_40.hdf5')
# 

# Fit the model
history = tiramisu.fit(train_data, train_label, callbacks=callbacks_list, batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, class_weight=class_weighting , validation_data=(test_data, test_label), shuffle=True) # validation_split=0.33

# This save the trained model weights to this file with number of epochs
tiramisu.save_weights('weights/tiramisu_model_weight_{}.hdf5'.format(nb_epoch))

