from .lrs_layer import LowRankSig_FirstOrder, LowRankSig_HigherOrder

import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import keras
from keras import backend as K
from keras import Input, Model
from keras.layers import Dense, BatchNormalization, Reshape, Concatenate

def init_blrs_model(input_shape, num_levels, num_hidden, num_classes, recursive_tensors=False, order=1):
    
    num_sig_layers = 3
    num_sig_hidden = num_hidden if recursive_tensors else int(num_hidden / num_levels)
    num_sig_hidden_per_direction = int(num_hidden / 2)
    
    inp = Input(input_shape)
    layer = inp
    
    for i in range(num_sig_layers-1):
        if order == 1:
            lrs_forward = LowRankSig_FirstOrder(num_sig_hidden_per_direction, num_levels, add_time=True, return_levels=True, return_sequences=True, reverse=False)(layer)
        else:
            lrs_forward = LowRankSig_HigherOrder(num_sig_hidden_per_direction, num_levels, add_time=True, return_levels=True, return_sequences=True, reverse=False, order=order)(layer)
        lrs_forward = Reshape((-1, num_sig_hidden_per_direction * num_levels))(lrs_forward)
        lrs_forward = BatchNormalization(axis=-1)(lrs_forward)
        
        if order == 1:
            lrs_reverse = LowRankSig_FirstOrder(num_sig_hidden_per_direction, num_levels, add_time=True, return_levels=True, return_sequences=True, reverse=True)(layer)
        else:
            lrs_reverse = LowRankSig_HigherOrder(num_sig_hidden_per_direction, num_levels, add_time=True, return_levels=True, return_sequences=True, reverse=True, order=order)(layer)
        lrs_reverse = Reshape((-1, num_sig_hidden_per_direction * num_levels))(lrs_reverse)
        lrs_reverse = BatchNormalization(axis=-1)(lrs_reverse)
        
        layer = Concatenate()([lrs_forward, lrs_reverse])
    
    if order == 1:
        lrs_final = LowRankSig_FirstOrder(num_sig_hidden, num_levels, add_time=True, return_levels=True)(layer)
    else:
        lrs_final = LowRankSig_HigherOrder(num_sig_hidden, num_levels, add_time=True, return_levels=True, order=order)(layer)
    lrs_final = Reshape((num_sig_hidden * num_levels, ))(lrs_final)
    lrs_final = BatchNormalization(axis=-1)(lrs_final)

    outp = Dense(num_classes, activation='softmax')(lrs_final)
    
    model = keras.Model(inp, outp)
    model.name = 'BLRS_M{}_H{}_D{}'.format(num_levels, num_hidden, order)

    return model