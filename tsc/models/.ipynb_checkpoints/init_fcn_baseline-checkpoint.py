import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import keras
from keras import backend as K
from keras.layers import Dense, BatchNormalization, Conv1D, Activation, GlobalAveragePooling1D, InputLayer

def init_fcn_baseline(input_shape, num_classes):
    
    model = keras.Sequential()

    model.add(InputLayer(input_shape=input_shape))
    
    model.add(Conv1D(128, 8, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv1D(256, 5, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv1D(128, 3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(GlobalAveragePooling1D())

    model.add(Dense(num_classes, activation='softmax'))
              
    model.name = 'FCN'

    return model