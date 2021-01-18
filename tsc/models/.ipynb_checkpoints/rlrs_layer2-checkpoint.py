import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import keras
from keras import backend as K
from keras import initializers, regularizers, layers, activations

# from tensorflow.python.training.tracking.data_structures import NoDependency

class RecurrentLRSCell2(layers.Layer):

    def __init__(self, units, num_levels, num_features, activation='tanh', use_projection=False, use_bias=False, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
                 projection_initializer='identity', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, projection_regularizer=None, bias_regularizer=None, **kwargs):             
        
        self.units = units
        self.num_features = num_features
        self.num_levels = num_levels
        self.activation = activations.get(activation)
        self.use_projection = use_projection
        self.use_bias = use_bias
        # self.state_size = NoDependency([self.num_levels * self.units, self.num_levels * self.units, self.num_features])
        self.state_size = [self.num_levels * self.units, self.num_levels * self.units, self.num_levels * self.units, self.num_features]
        self.output_size = self.num_levels * self.units
        
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.projection_initializer = initializers.get(projection_initializer)
        
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.projection_regularizer = regularizers.get(projection_regularizer)
        
        super(RecurrentLRSCell2, self).__init__(**kwargs)

    def build(self, input_shape):
        assert self.num_features == input_shape[-1]
        self.kernel = self.add_weight(shape=(self.num_features,  self.num_levels * self.units),
                                      initializer=self.kernel_initializer, regularizer=self.kernel_regularizer, name='kernel')
        self.recurrent_kernel = self.add_weight(shape=(self.num_levels * self.units, self.num_levels * self.units),
                                                initializer=self.recurrent_initializer, regularizer=self.recurrent_regularizer, name='recurrent_kernel')
        
        if self.use_projection:
            self.projection_kernel = self.add_weight(shape=(self.num_levels * self.units, self.num_levels * self.units),
                                                     initializer=self.projection_initializer, regularizer=self.projection_regularizer, name='projection_kernel')
    
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.num_levels * self.units,),
                                                    initializer=self.bias_initializer, regularizer=self.bias_regularizer, name='bias')
        self.built = True

    def call(self, inputs, states):
        num_samples = tf.shape(inputs)[0]
        xt = inputs
        ht_1 = states[0]
        ht_2 = states[1]
        st_1 = states[2]
        xt_1 = states[3] 
        
        st = st_1 + tf.concat((tf.ones((num_samples, self.units), dtype=xt.dtype), st_1[:, :-self.units]), axis=1) * (tf.matmul(xt-xt_1, self.kernel) + tf.matmul(ht_1-ht_2, self.recurrent_kernel))
        
        ht = st
        
        if self.use_projection:
            ht = tf.matmul(ht, self.projection_kernel)
        
        if self.use_bias:
            ht += self.bias[None, :]
        
        if self.activation is not None:
            ht = self.activation(ht)
        
        return ht, [ht, ht_1, st, xt]
    
    def get_config(self):
        config = {
            'units': self.units,
            'num_features' : self.num_features,
            'num_levels' : self.num_levels,
            'activation' : activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer)
        }
        base_config = super(RecurrentLRSCell2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))