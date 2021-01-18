from .lrs_layer import LowRankSig_FirstOrder, LowRankSig_HigherOrder

import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import keras
from keras import backend as K
from keras.layers import Dense, BatchNormalization, Reshape, Conv1D, ZeroPadding1D, Lambda, Activation

def init_convlrs2_model(input_shape, num_levels, num_hidden, num_classes, recursive_tensors=False, reverse=False, order=1, activation=None):
    
    activation = activation or 'relu'
    num_sig_layers = 3
    num_sig_hidden = num_hidden if recursive_tensors else int(num_hidden / num_levels)
    
    model = keras.Sequential()

    model.add(keras.layers.InputLayer(input_shape=input_shape))
    
    model.add(Conv1D(num_hidden, 8, padding='causal'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation(activation))
    
    model.add(Conv1D(num_hidden, 5, padding='causal'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation(activation))
    
    model.add(Conv1D(num_hidden, 3, padding='causal'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation(activation))
    
    for i in range(num_sig_layers-1):
        if order == 1:
            model.add(LowRankSig_FirstOrder(num_sig_hidden, num_levels, add_time=True, return_levels=True, return_sequences=True, reverse=reverse, recursive_tensors=recursive_tensors))
        else:
            model.add(LowRankSig_HigherOrder(num_sig_hidden, num_levels, add_time=True, return_levels=True, return_sequences=True, reverse=reverse, recursive_tensors=recursive_tensors, order=order))
        model.add(Reshape((-1, num_sig_hidden * num_levels,)))
        model.add(BatchNormalization(axis=-1))
    
    if order == 1:
        model.add(LowRankSig_FirstOrder(num_sig_hidden, num_levels, add_time=True, return_levels=True, reverse=reverse, recursive_tensors=recursive_tensors))
    else:
        model.add(LowRankSig_HigherOrder(num_sig_hidden, num_levels, add_time=True, return_levels=True, reverse=reverse, recursive_tensors=recursive_tensors, order=order))
    model.add(Reshape((num_sig_hidden * num_levels,)))
    model.add(BatchNormalization(axis=-1))
    
    model.add(Dense(num_classes, activation='softmax'))
              
    model.name = 'ConvLRS2_M{}_H{}_D{}'.format(num_levels, num_hidden, order) if activation == 'relu' else 'ConvLRS2Tanh_M{}_H{}_D{}'.format(num_levels, num_hidden, order)

    return model