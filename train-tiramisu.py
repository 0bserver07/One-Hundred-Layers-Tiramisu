from __future__ import absolute_import
from __future__ import print_function
import os


import keras.models as models
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, Cropping2D
from keras.layers.normalization import BatchNormalization

from keras.layers import Conv2D, Conv2DTranspose

from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, merge
from keras.regularizers import l2
from keras.models import Model
from keras import regularizers

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D

from keras import backend as K

from keras import callbacks
import math
# remote = callbacks.RemoteMonitor(root='http://localhost:9000', path='/publish/epoch/end/', field='data', headers=None)

# early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=0, mode='auto')

# tensor_board = callbacks.TensorBoard(log_dir='./logs', histogram_freq=5, write_graph=True, write_images=True)


K.set_image_dim_ordering('tf')

import cv2
import numpy as np
import json


np.random.seed(7) # 0bserver07 for reproducibility




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


# load the data
train_data = np.load('./data/train_data.npy')
train_data = train_data.reshape((367, 224, 224, 3))

train_label =  np.load('./data/train_label.npy')#[:,:,:-1]



test_data = np.load('./data/test_data.npy')
test_data = test_data.reshape((233, 224, 224, 3))


test_label = np.load('./data/test_label.npy')#[:,:,:-1]

# test_label = to_categorical(test_label, num_classes=None)

# load the model:
with open('tiramisu_fc_dense67_model_12_func.json') as model_file:
    tiramisu = models.model_from_json(model_file.read())


# section 4.1 from the paper

from keras.callbacks import LearningRateScheduler
 
# learning rate schedule
def step_decay(epoch):
	initial_lrate = 0.001
	drop = 0.00001
	epochs_drop = 10.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

lrate = LearningRateScheduler(step_decay)

# tiramisu.load_weights("weights/prop_tiramisu_weights_67_12_func_10-e5_decay.best.hdf5")




optimizer = RMSprop(lr=0.001, decay=0.0000001)
# optimizer = SGD(lr=0.01)
# optimizer = Adam(lr=1e-3, decay=0.995)

tiramisu.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# learning schedule callback
# lrate = LearningRateScheduler(step_decay)

# checkpoint 278
TensorBoard = callbacks.TensorBoard(log_dir='./logs', histogram_freq=5, write_graph=True, write_images=True)
# ReduceLROnPlateau = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

filepath="weights/prop_tiramisu_weights_67_12_func_10-e7_decay.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=2,
									 save_best_only=True, save_weights_only=False, mode='max')


callbacks_list = [checkpoint]

nb_epoch = 150
batch_size = 2


# Fit the model
history = tiramisu.fit(train_data, train_label, batch_size=batch_size, epochs=nb_epoch,
	callbacks=callbacks_list, class_weight=class_weighting,verbose=1, validation_data=(test_data, test_label), shuffle=True) # validation_split=0.33




# This save the trained model weights to this file with number of epochs
tiramisu.save_weights('weights/prop_tiramisu_weights_67_12_func_10-e7_decay{}.hdf5'.format(nb_epoch))

import matplotlib.pyplot as plt
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
