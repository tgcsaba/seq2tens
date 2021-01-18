from .layers import Time, Difference, LRS

import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import keras
from keras import backend as K
from keras.layers import Dense, BatchNormalization, Reshape, Conv1D, ZeroPadding1D, Lambda, Activation

def init_convlrs_model(input_shape, num_levels, num_hidden, num_classes, difference=True, add_time=True, recursive_tensors=True, reverse=False):
    
    num_sig_layers = 3
    num_sig_hidden = num_hidden
    
    model = keras.Sequential()

    model.add(keras.layers.InputLayer(input_shape=input_shape))
    
    if add_time:
        model.add(Time())
    model.add(Conv1D(num_hidden, 8, padding='same', kernel_initializer='he_uniform'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    
    if add_time:
        model.add(Time())
    model.add(Conv1D(num_hidden, 5, padding='same',  kernel_initializer='he_uniform'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    
    if add_time:
        model.add(Time())
    model.add(Conv1D(num_hidden, 3, padding='same',  kernel_initializer='he_uniform'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    
    for i in range(num_sig_layers-1):
        if add_time:
            model.add(Time())
        if difference:
            model.add(Difference())
        model.add(LRS(num_sig_hidden, num_levels, return_sequences=True, reverse=reverse, recursive_tensors=recursive_tensors))
        model.add(Reshape((-1, num_sig_hidden * num_levels,)))
        model.add(BatchNormalization(axis=-1))
    
    if add_time:
        model.add(Time())
    if difference:
        model.add(Difference())
    model.add(LRS(num_sig_hidden, num_levels, reverse=reverse, recursive_tensors=recursive_tensors))
    model.add(Reshape((num_sig_hidden * num_levels,)))
    model.add(BatchNormalization(axis=-1))
    
    model.add(Dense(num_classes, activation='softmax'))
    
    diff_tag = 'D' if difference else ''
    
    model.name = 'Conv{}LRS_M{}_H{}'.format(diff_tag, num_levels, num_hidden)

    return model