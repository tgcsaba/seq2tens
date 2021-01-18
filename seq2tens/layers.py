import numpy as np
import tensorflow as tf

from .initializers import LS2TUniformInitializer, LS2TNormalInitializer
from .algorithms import low_rank_seq2tens

from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras.layers import Layer


class Time(Layer):

    def __init__(self, **kwargs):
        super(Time, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) == 3        
        super(Time, self).build(input_shape)
        
    def call(self, X, mask=None):
        
        if mask is not None:            
            mask_float = tf.cast(mask, X.dtype)
            time = (tf.cumsum(mask_float, axis=1, exclusive=True) / (tf.reduce_sum(mask_float, axis=1)[:, None] - 1.) * 2. - mask_float)[..., None]
            X = tf.concat((time, X), axis=-1)
        else:
            num_examples = tf.shape(X)[0]
            time = tf.tile(tf.range(tf.cast(tf.shape(X)[1], X.dtype), dtype=X.dtype)[None, :, None], [num_examples, 1, 1])
            time *= 2. / (tf.cast(tf.shape(X)[1], X.dtype) - 1.)
            time -= 1.
            X = tf.concat((time, X), axis=-1)
        return X 
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2] + 1)

class Difference(Layer):
    
    def __init__(self, **kwargs):
        super(Difference, self).__init__(**kwargs)
        self.supports_masking = True
    
    def build(self, input_shape):
        assert len(input_shape) == 3        
        super(Difference, self).build(input_shape)
    
    def call(self, X, mask=None):
        if mask is not None:
            X = tf.where(mask[..., None], X, tf.zeros_like(X))
        X = tf.concat((tf.zeros_like(X[:, 0])[:, None], X), axis=1)
        return X[:, 1:] - X[:, :-1]
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
class LS2T(Layer):

    def __init__(self, 
                 num_functionals,
                 num_levels,
                 embedding_order=1,
                 recursive_weights=False, 
                 reverse=False,
                 use_bias=True,
                 kernel_initializer='ls2t_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 return_sequences=False,
                 **kwargs):
    
    
        super(LS2T, self).__init__(activity_regularizer=activity_regularizer, **kwargs)
        
        self.num_functionals = num_functionals
        self.num_levels = num_levels
        self.embedding_order = embedding_order
        
        self.recursive_weights = recursive_weights
        self.use_bias = use_bias
        self.reverse = reverse        
        self.return_sequences = return_sequences
        
        if kernel_initializer.lower().replace('_', '') == 'ls2tuniform':
            self.kernel_initializer = LS2TUniformInitializer(num_levels, recursive_weights)
        elif kernel_initializer.lower().replace('_', '') == 'ls2tnormal':
            self.kernel_initializer = LS2TNormalInitializer(num_levels, recursive_weights)
        else:
            self.kernel_initializer = initializers.get(kernel_initializer)
            
        self.bias_initializer = initializers.get(bias_initializer)
        
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
    
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        
        self.supports_masking = True        

    def build(self, input_shape):
        self.num_features = input_shape.as_list()[-1]
        
        assert len(input_shape) == 3
        
        self.num_components = int(self.num_levels * (self.num_levels+1) / 2.) if not self.recursive_weights else self.num_levels
        
        self.kernel = self.add_weight(shape=tf.TensorShape([self.num_components, self.num_features, self.num_functionals]),
                                      name='kernel',
                                      initializer=self.kernel_initializer, 
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      dtype=self.dtype,
                                      trainable=True)
        
        self.bias = self.add_weight(shape=tf.TensorShape([self.num_components, self.num_functionals]),
                                    name='bias',
                                    initializer=self.bias_initializer,
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint,
                                    dtype=self.dtype,
                                    trainable=True)
        
        super(LS2T, self).build(input_shape)
        
    def call(self, X, mask=None):
        return low_rank_seq2tens(X, self.kernel, self.num_levels, embedding_order=self.embedding_order, recursive_weights=self.recursive_weights,
                                 bias=self.bias, reverse=self.reverse, return_sequences=self.return_sequences, mask=mask)

    def compute_output_shape(self, input_shape):
        if self.return_sequences:
            return (input_shape[0], input_shape[1], self.num_levels, self.num_functionals)
        else:
            return (input_shape[0], self.num_levels, self.num_functionals)
