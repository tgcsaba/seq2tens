import numpy as np
import tensorflow as tf

from .initializers import LS2TInitializer

from tensorflow.keras import initializers, layers


class Time(layers.Layer):

    def __init__(self, mask=False, **kwargs):
        self.mask = mask
        super(Time, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3        
        super(Time, self).build(input_shape)
        
    def call(self, X):
        
        if self.mask:
            mask = tf.logical_not(tf.reduce_all(tf.equal(X, 0.), axis=-1))
            mask = tf.logical_and(mask, tf.logical_not(tf.reduce_all(tf.equal(tf.concat((tf.zeros_like(X[:, :1], dtype=X.dtype), X[:, 1:] - X[:, :-1]), axis=1), 0.), axis=-1)))
            
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

class Difference(layers.Layer):
    
    def __init__(self, **kwargs):
        super(Difference, self).__init__(**kwargs)
    
    def build(self, input_shape):
        super(Difference, self).build(input_shape)
    
    def call(self, X, mask=None):
        if mask is not None:
            X = tf.where(mask, X, tf.zeros_like(X))
        X = tf.concat((tf.zeros_like(X[:, 0:1]), X), axis=1)
        X = X[:, 1:] - X[:, :-1]
        return X
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
class LS2T(layers.Layer):

    def __init__(self, units, num_levels, use_bias=True, recursive_tensors=False, reverse=False, return_sequences=False, **kwargs):
        
        self.units = units
        self.num_levels = num_levels
        
        self.use_bias = use_bias
        self.recursive_tensors = recursive_tensors
        self.reverse = reverse
        
        self.return_sequences = return_sequences
        
        super(LS2T, self).__init__(**kwargs)

    def build(self, input_shape):
        self.num_features = input_shape.as_list()[-1]
        assert len(input_shape) == 3
        
        self.len_tensors = int(self.num_levels * (self.num_levels+1) / 2.) if not self.recursive_tensors else self.num_levels
        
        self.kernel = self.add_weight(shape=tf.TensorShape([self.len_tensors, self.num_features, self.units]),
                                      name='kernel', initializer=LS2TInitializer(self.num_levels, self.recursive_tensors), trainable=True)
        
        self.bias = self.add_weight(shape=tf.TensorShape([self.len_tensors, self.units]),
                                    name='bias', initializer=initializers.Zeros(), trainable=True)
        
        super(LS2T, self).build(input_shape)
        
    def call(self, X, mask=None):
                
        num_examples, len_examples = tf.shape(X)[0], tf.shape(X)[1]
        
        M = tf.matmul(tf.reshape(X, [1, -1, self.num_features]), self.kernel)
        
        M = tf.reshape(M, [self.len_tensors, num_examples, len_examples, self.units]) + self.bias[:, None, None, :]
        
        if mask is not None:
            M = tf.where(mask[None, :, :, None], M, tf.zeros_like(M))
        
        if self.return_sequences:
            Y = [tf.cumsum(M[0], reverse=self.reverse, axis=1)]
        else:
            Y = [tf.reduce_sum(M[0], axis=1)]
        
        if not self.recursive_tensors:
            k = 1
            for m in range(1, self.num_levels):
                R = M[k]
                k += 1
                for i in range(1, m+1):
                    R = M[k] *  tf.cumsum(R, reverse=self.reverse, exclusive=True, axis=1)
                    k += 1
                if self.return_sequences:
                    Y.append(tf.cumsum(R, reverse=self.reverse, axis=1))
                else:
                    Y.append(tf.reduce_sum(R, axis=1))
        else:
            R = M[0]
            for m in range(1, self.num_levels):
                R = M[m] * tf.cumsum(R, exclusive=True, reverse=self.reverse, axis=1)
                
                if self.return_sequences:
                    Y.append(tf.cumsum(R, reverse=self.reverse, axis=1))
                else:
                    Y.append(tf.reduce_sum(R, axis=1))
        
        return tf.stack(Y, axis=-2)

    def compute_output_shape(self, input_shape):
        if self.return_sequences:
            return (input_shape[0], input_shape[1], self.num_levels, self.units)
        else:
            return (input_shape[0], self.num_levels, self.units)
