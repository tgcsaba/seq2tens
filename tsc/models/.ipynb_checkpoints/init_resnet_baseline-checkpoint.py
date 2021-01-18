import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import keras
from keras import backend as K
from keras import Input, Model
from keras.layers import Dense, BatchNormalization, Conv1D, Activation, GlobalAveragePooling1D, Add, InputLayer

def init_resnet_baseline(input_shape, num_classes):
    
    num_hidden = 64
    
    inp = Input(input_shape)
    
    # block 1
    
    conv_x = Conv1D(num_hidden, 8, padding='same')(inp)
    conv_x = BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)
    
    conv_y = Conv1D(num_hidden, 5, padding='same')(conv_x)
    conv_y = BatchNormalization()(conv_y)
    conv_y = Activation('relu')(conv_y)
    
    conv_z = Conv1D(num_hidden, 3, padding='same')(conv_y)
    conv_z = BatchNormalization()(conv_z)
    
    ## expand channels
    shortcut_1 = Conv1D(num_hidden, 1, padding='same')(inp)
    shortcut_1 = BatchNormalization()(shortcut_1)

    outp_1 = Add()([shortcut_1, conv_z])
    outp_1 = Activation('relu')(outp_1)
    
    # block 2
    
    conv_x = Conv1D(2 * num_hidden, 8, padding='same')(outp_1)
    conv_x = BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)

    conv_y = Conv1D(2 * num_hidden, 5, padding='same')(conv_x)
    conv_y = BatchNormalization()(conv_y)
    conv_y = Activation('relu')(conv_y)

    conv_z = Conv1D(2 * num_hidden, 3, padding='same')(conv_y)
    conv_z = BatchNormalization()(conv_z)

    ## expand channels
    shortcut_2 = Conv1D(2 * num_hidden, 1, padding='same')(outp_1)
    shortcut_2 = BatchNormalization()(shortcut_2)

    outp_2 = Add()([shortcut_2, conv_z])
    outp_2 = Activation('relu')(outp_2)
    
    # block 3
    
    conv_x = keras.layers.Conv1D(2 * num_hidden, 8, padding='same')(outp_2)
    conv_x = BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)

    conv_y = Conv1D(2 * num_hidden, 5, padding='same')(conv_x)
    conv_y = BatchNormalization()(conv_y)
    conv_y = Activation('relu')(conv_y)

    conv_z = Conv1D(2 * num_hidden, 3, padding='same')(conv_y)
    conv_z = BatchNormalization()(conv_z)

    ## channels are equal now
    shortcut_y = BatchNormalization()(outp_2)

    outp_3 = Add()([shortcut_2, conv_z])
    outp_3 = Activation('relu')(outp_3)
    
    # pooling and prediction layer
    gap = GlobalAveragePooling1D()(outp_3)
    outp = Dense(num_classes, activation='softmax')(gap)

    model = Model(inp, outp)
    model.name = 'ResNet'

    return model