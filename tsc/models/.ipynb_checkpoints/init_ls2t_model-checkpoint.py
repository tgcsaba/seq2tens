import sys

sys.path.append('..')

# from .layers import Time, Difference, LS2T

import seq2tens

import numpy as np
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import InputLayer, Conv1D, Activation, Reshape, Dense, LayerNormalization, BatchNormalization, Lambda, GlobalAveragePooling1D, TimeDistributed, Masking

def init_ls2t_model(preprocess_size, ls2t_size, ls2t_order, ls2t_depth, preprocess='conv', preprocess_time=True, ls2t_diff=True, ls2t_time=True, recursive_tensors=True, name_only=False, input_shape=None, num_classes=None):
    
    preprocess = preprocess.lower()
    
    if preprocess != 'conv' and preprocess != 'dense':
        preprocess_name = ''
    else:
        preprocess_name = 'Conv' if preprocess == 'conv' else 'Dense'
    
    
    ls2t_recurrent_tag = 'R' if recursive_tensors else ''
    ls2t_name = 'LS2T{}'.format(ls2t_recurrent_tag)
    
    model_name = '{}{}_H{}_N{}_M{}_D{}'.format(preprocess_name, ls2t_name, preprocess_size, ls2t_size, ls2t_order, ls2t_depth)
    
    if name_only: return model_name
    
    
    model = Sequential()

    model.add(InputLayer(input_shape=input_shape))
    model.add(Masking(mask_value=0.))
    
    if preprocess_name != '':
        if preprocess_time:
            model.add(seq2tens.layers.Time())
        if preprocess == 'conv':
            model.add(Conv1D(preprocess_size, 8, padding='same', kernel_initializer='he_uniform'))
        else:
            model.add(TimeDistributed(Dense(preprocess_size, kernel_initializer='he_uniform')))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))


        if preprocess_time:
            model.add(seq2tens.layers.Time())
        if preprocess == 'conv':
            model.add(Conv1D(preprocess_size, 5, padding='same' , kernel_initializer='he_uniform'))
        else:
            model.add(TimeDistributed(Dense(preprocess_size, kernel_initializer='he_uniform')))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))


        if preprocess_time:
            model.add(seq2tens.layers.Time())
        if preprocess == 'conv':
            model.add(Conv1D(preprocess_size, 3, padding='same', kernel_initializer='he_uniform'))
        else:
            model.add(Dense(preprocess_size, kernel_initializer='he_uniform'))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))
    
    
    for i in range(ls2t_depth-1):
        if ls2t_time:
            model.add(seq2tens.layers.Time())
        if ls2t_diff:
            model.add(seq2tens.layers.Difference())
        model.add(seq2tens.layers.LS2T(ls2t_size, ls2t_order, return_sequences=True, recursive_tensors=recursive_tensors))
        model.add(Reshape((input_shape[0], ls2t_order, ls2t_size,)))
        model.add(BatchNormalization(axis=[-2]))
        model.add(Reshape((input_shape[0], ls2t_order * ls2t_size,)))

    if ls2t_time:
        model.add(seq2tens.layers.Time())
    if ls2t_diff:
        model.add(seq2tens.layers.Difference())
    model.add(seq2tens.layers.LS2T(ls2t_size, ls2t_order, return_sequences=False, recursive_tensors=recursive_tensors))
    model.add(Reshape((ls2t_order, ls2t_size,)))
    model.add(BatchNormalization(axis=1))
    model.add(Reshape((ls2t_order * ls2t_size,)))
    
    model.add(Dense(num_classes, activation='softmax'))

    
    model._name = model_name

    return model
