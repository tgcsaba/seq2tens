import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import keras
from keras import backend as K
from keras import initializers, regularizers, layers

class LowRankSig_FirstOrder(layers.Layer):

    def __init__(self, units, num_levels, difference=True, add_time=True, window=None, return_sequences=False, kernel_initializer='glorot_uniform', kernel_regularizer=None,
                 recursive_tensors=False, reverse=False, return_levels=False, second_difference=False, **kwargs):
        
        self.units = units
        self.num_levels = num_levels
        self.difference = difference
        self.second_difference = second_difference
        self.add_time = add_time
        self.window = window
        self.recursive_tensors = recursive_tensors
        self.return_levels = return_levels
        self.reverse = reverse
        
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        
        self.return_sequences = return_sequences
        
        super(LowRankSig_FirstOrder, self).__init__(**kwargs)

    def build(self, input_shape):
        
        assert len(input_shape) == 3
        
        self.num_features = input_shape[-1] + int(self.add_time)
        self.len_examples = input_shape[-2]
        
        self.len_tensors = int(self.num_levels * (self.num_levels+1) / 2.) if not self.recursive_tensors else self.num_levels
        
        self.kernel = self.add_weight(shape=(self.num_features, self.len_tensors, self.units),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      trainable=True)
        
        super(LowRankSig_FirstOrder, self).build(input_shape)
        
    def call(self, X):
        
        num_examples = tf.shape(X)[0]
        if self.add_time:
            time = tf.tile(tf.range(tf.cast(tf.shape(X)[1], X.dtype), dtype=X.dtype)[None, :, None], [num_examples, 1, 1])
            time *= 2. / (tf.cast(tf.shape(X)[1], X.dtype) - 1.)
            time -= 1.
            X = tf.concat((X,  time), axis=-1)
        
        M = tf.matmul(tf.reshape(X, [-1, self.num_features]), tf.reshape(self.kernel, [self.num_features, -1]))
        M = tf.reshape(M, [-1, self.len_examples, self.len_tensors, self.units])
        
        if self.difference:
            M = tf.concat((tf.zeros_like(M[:, :1]), M[:, 1:] - M[:, :-1]), axis=1)
        
        if self.second_difference:
            M = tf.concat((M, tf.concat((tf.zeros_like(M[:, 0][:, None]), M[:, 1:] - M[:, :-1]), axis=1)), axis=2)
        
        if self.return_sequences:
            Y = [tf.cumsum(M[..., 0, :], reverse=self.reverse, axis=1)]
        else:
            Y = [tf.reduce_sum(M[..., 0, :], axis=1)]
        
        if not self.recursive_tensors:
            k = 1
            for m in range(2, self.num_levels+1):
                R = M[..., k, :]
                k += 1
                for i in range(1, m):
                    R = M[..., k, :] *  tf.cumsum(R, reverse=self.reverse, exclusive=True, axis=1)
                    k += 1
                if self.return_sequences:
                    Y.append(tf.cumsum(R, reverse=self.reverse, axis=1))
                else:
                    Y.append(tf.reduce_sum(R, axis=1))
        else:
            R = M[..., 0, :]
            for m in range(1, self.num_levels):
                R = M[..., m, :] * tf.cumsum(R, exclusive=True, reverse=self.reverse, axis=1)
                
                if self.return_sequences:
                    Y.append(tf.cumsum(R, reverse=self.reverse, axis=1))
                else:
                    Y.append(tf.reduce_sum(R, axis=1))
        if self.return_levels:
            return tf.stack(Y, axis=-2)
        else:
            return tf.add_n(Y)

    def compute_output_shape(self, input_shape):
        if self.return_sequences:
            if self.return_levels:
                return (input_shape[0], self.len_examples - int(self.difference), self.num_levels, self.units)
            else:
                return (input_shape[0], self.len_examples - int(self.difference), self.units)
        else:
            if self.return_levels:
                return (input_shape[0], self.num_levels, self.units)
            else:
                return (input_shape[0], self.units)