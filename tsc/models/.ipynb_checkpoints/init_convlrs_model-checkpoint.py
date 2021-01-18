from .layers import Time, Difference, LRS

import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import InputLayer, Conv1D, Activation, Reshape, Dense, LayerNormalization, BatchNormalization, Lambda, GlobalAveragePooling1D

def init_convlrs_model(conv_size, lrs_size, lrs_levels, conv_time=False, lrs_diff=True, lrs_time=True, recursive_tensors=True, name_only=False, input_shape=None, num_classes=None, lrs_norm='BN'):
    
    conv_name = 'TConv' if conv_time else 'Conv'
    
    lrs_time_tag = 'T' if lrs_time else ''
    lrs_diff_tag = 'D' if lrs_diff else ''
    lrs_recurrent_tag = 'R' if recursive_tensors else ''
    lrs_name = '{}{}{}LRS{}'.format(lrs_time_tag, lrs_diff_tag, lrs_recurrent_tag, lrs_norm)
    
    model_name = '{}{}_H{}_W{}'.format(conv_name, lrs_name, conv_size, lrs_size)
    
    if name_only: return model_name
    
    num_sig_layers = 3
    
    model = Sequential()

    model.add(InputLayer(input_shape=input_shape))
    
    if conv_time:
        model.add(Time())
    model.add(Conv1D(conv_size, 8, padding='same', kernel_initializer='he_uniform'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    
    
    if conv_time:
        model.add(Time())
    model.add(Conv1D(conv_size, 5, padding='same' , kernel_initializer='he_uniform'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    
    
    if conv_time:
        model.add(Time())
    model.add(Conv1D(conv_size, 3, padding='same', kernel_initializer='he_uniform'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    
    
    for i in range(num_sig_layers-1):
#         model.add(Conv1D(conv_size, 3, padding='same', kernel_initializer='he_uniform'))
#         model.add(BatchNormalization(axis=-1))
        if lrs_time:
            model.add(Time())
        if lrs_diff:
            model.add(Difference())
        model.add(LRS(lrs_size, lrs_levels, return_sequences=True, recursive_tensors=recursive_tensors))
        model.add(Reshape((input_shape[0]-i-1, lrs_levels, lrs_size,)))
        if lrs_norm == 'BN':
            model.add(BatchNormalization(axis=[-2]))
        elif lrs_norm == 'TBN':
            model.add(BatchNormalization(axis=[-2, -1]))
        elif lrs_norm == 'LN':
            model.add(LayerNormalization(axis=[-3, -1]))
#         model.add(LayerNormalization(axis=[1, 3]))
#model.add(LayerNormalization(axis=[1, 3]))
        model.add(Reshape((input_shape[0]-i-1, lrs_levels * lrs_size,)))
#         model.add(Activation('relu'))
#         if renorm:
#             model.add(BatchNormalization(axis=[1, 2], renorm=True, center=False, scale=False))
#         else:
        
#         
#     model.add(Conv1D(conv_size, 3, padding='same', kernel_initializer='he_uniform'))
#     model.add(BatchNormalization(axis=-1))
    if lrs_time:
        model.add(Time())
    if lrs_diff:
        model.add(Difference())
    model.add(LRS(lrs_size, lrs_levels, return_sequences=False, recursive_tensors=recursive_tensors))
#     model.add(BatchNormalization(axis=1, center=False, scale=False))
    model.add(Reshape((lrs_levels, lrs_size,)))
#     model.add(LayerNormalization(axis=[2]))
    model.add(BatchNormalization(axis=1))
    model.add(Reshape((lrs_levels * lrs_size,)))
#     model.add(BatchNormalization(axis=-1, renorm=renorm, center=False, scale=False))

#     model.add(tf.keras.layers.Dropout(0.5))
#     if gap:
#         model.add(GlobalAveragePooling1D())
#     else:
#     model.add(Lambda(lambda X: X[:, -1]))
    
    model.add(Dense(num_classes, activation='softmax'))
    
    model._name = model_name

    return model