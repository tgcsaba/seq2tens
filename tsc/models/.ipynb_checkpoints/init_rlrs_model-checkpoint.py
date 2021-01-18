from .lrs_layer import TimeAugmentation
from .rlrs_layer import RecurrentLRS

import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import keras
from keras import backend as K
from keras.layers import Dense, BatchNormalization, Reshape

def init_rlrs_model(input_shape, num_levels, num_hidden, num_classes, decoupled=False):
    
    model = keras.Sequential()

    model.add(keras.layers.InputLayer(input_shape=input_shape))
    
    model.add(TimeAugmentation())
    model.add(RecurrentLRS(num_hidden, num_levels, input_shape[-1] + 1, decoupled=decoupled))
    model.add(BatchNormalization(axis=-1))
    
    model.add(Dense(num_classes, activation='softmax'))
    
    if not decoupled:
        model.name = 'RLRS_M{}_H{}'.format(num_levels, num_hidden) 
    else:
        model.name = 'RLRS2_M{}_H{}'.format(num_levels, num_hidden) 

    return model