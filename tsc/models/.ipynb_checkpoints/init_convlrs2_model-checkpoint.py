from .layers import Time, Difference, LRS

import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import InputLayer, Conv1D, Activation, Reshape, Dense, LayerNormalization, BatchNormalization

def init_convlrs2_model(input_shape, num_levels, num_hidden, num_classes, difference=True, add_time=True, recursive_tensors=True, reverse=False, layer_norm=False):
    
    num_sig_layers = 3
    num_sig_hidden = int(num_hidden / num_levels)
#     num_levels = 3
    
    model = Sequential()

    model.add(InputLayer(input_shape=input_shape))
    
    if add_time:
        model.add(Time())
    model.add(Conv1D(num_hidden, 8, padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    
    if add_time:
        model.add(Time())
    model.add(Conv1D(num_hidden, 5, padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    
    if add_time:
        model.add(Time())
    model.add(Conv1D(num_hidden, 3, padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    
    for i in range(num_sig_layers-1):
        if add_time:
            model.add(Time())
        if difference:
            model.add(Difference())
        model.add(LRS(num_sig_hidden, num_levels, return_sequences=True, reverse=reverse, recursive_tensors=recursive_tensors))
        model.add(Reshape((input_shape[0], num_levels, num_sig_hidden,)))
        if layer_norm:
            model.add(LayerNormalization(axis=-1, center=False, scale=False))
        else:
            model.add(BatchNormalization(axis=[1, 2], center=False, scale=False))
        model.add(Reshape((input_shape[0], num_sig_hidden * num_levels,)))
    
    if add_time:
        model.add(Time())
    if difference:
        model.add(Difference())
    model.add(LRS(num_sig_hidden, num_levels, reverse=reverse, recursive_tensors=recursive_tensors))
    model.add(Reshape((num_sig_hidden, num_levels,)))
    if layer_norm:
        model.add(LayerNormalization(axis=-1, center=False, scale=False))
    else:
        model.add(BatchNormalization(axis=1, center=False, scale=False))
    model.add(Reshape((num_sig_hidden * num_levels,)))
    
    model.add(Dense(num_classes, activation='softmax'))
    
    diff_tag = 'D' if difference else ''
    norm_tag = 'LN' if layer_norm else 'BN'
    
    model._name = 'Conv{}LRS{}2_M{}_H{}'.format(diff_tag, norm_tag, num_levels, num_hidden)

    return model