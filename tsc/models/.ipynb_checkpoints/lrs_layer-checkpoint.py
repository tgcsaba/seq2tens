import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import keras
from keras import backend as K
from keras import initializers, regularizers, layers

class TimeAugmentation(layers.Layer):

    def __init__(self, **kwargs):
        super(TimeAugmentation, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3        
        super(TimeAugmentation, self).build(input_shape)
        
    def call(self, X):
        num_examples = tf.shape(X)[0]
        time = tf.tile(tf.range(tf.cast(tf.shape(X)[1], X.dtype), dtype=X.dtype)[None, :, None], [num_examples, 1, 1])
        time *= 2. / (tf.cast(tf.shape(X)[1], X.dtype) - 1.)
        time -= 1.
        X = tf.concat((time, X), axis=-1)
        return X
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2]+1)

class LowRankSig_FirstOrder(layers.Layer):

    def __init__(self, units, num_levels, difference=True, add_time=True, return_sequences=False, kernel_initializer='glorot_uniform', kernel_regularizer=None,
                 recursive_tensors=False, reverse=False, return_levels=False, **kwargs):
        
        self.units = units
        self.num_levels = num_levels
        self.difference = difference
        self.add_time = add_time
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
            X = tf.concat((time, X), axis=-1)
        
        M = tf.matmul(tf.reshape(X, [-1, self.num_features]), tf.reshape(self.kernel, [self.num_features, -1]))
        M = tf.reshape(M, [-1, self.len_examples, self.len_tensors, self.units])
        
        # do final differencing
        if self.difference:
            M = tf.concat((tf.zeros_like(M[:, :1]), M[:, 1:] - M[:, :-1]), axis=1)
        
        if self.return_sequences:
            Y = [tf.cumsum(M[..., 0, :], reverse=self.reverse, axis=1)]
        else:
            Y = [tf.reduce_sum(M[..., 0, :], axis=1)]
        
        if not self.recursive_tensors:
            k = 1
            for m in range(1, self.num_levels):
                R = M[..., k, :]
                k += 1
                for i in range(1, m+1):
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
                return (input_shape[0], self.len_examples, self.num_levels, self.units)
            else:
                return (input_shape[0], self.len_examples, self.units)
        else:
            if self.return_levels:
                return (input_shape[0], self.num_levels, self.units)
            else:
                return (input_shape[0], self.units)
            
class LowRankSig_HigherOrder(layers.Layer):

    def __init__(self, units, num_levels, order=-1, difference=True, add_time=True, return_sequences=False, kernel_initializer='glorot_uniform', kernel_regularizer=None,
                 recursive_tensors=False, reverse=False, return_levels=False, **kwargs):
        
        self.units = units
        self.num_levels = num_levels
        self.order = num_levels if order < 1 else order
        self.difference = difference
        self.add_time = add_time
        self.recursive_tensors = recursive_tensors
        self.return_levels = return_levels
        self.reverse = reverse
        
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        
        self.return_sequences = return_sequences
        
        super(LowRankSig_HigherOrder, self).__init__(**kwargs)

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
        
        super(LowRankSig_HigherOrder, self).build(input_shape)
        
    def call(self, X):
        
        num_examples = tf.shape(X)[0]
        
        if self.add_time:
            time = tf.tile(tf.range(tf.cast(tf.shape(X)[1], X.dtype), dtype=X.dtype)[None, :, None], [num_examples, 1, 1])
            time *= 2. / (tf.cast(tf.shape(X)[1], X.dtype) - 1.)
            time -= 1.
            X = tf.concat((time, X), axis=-1)
        
        M = tf.matmul(tf.reshape(X, [-1, self.num_features]), tf.reshape(self.kernel, [self.num_features, -1]))
        M = tf.reshape(M, [-1, self.len_examples, self.len_tensors, self.units])
        
        # do final differencing
        if self.difference:
            M = tf.concat((tf.zeros_like(M[:, :1]), M[:, 1:] - M[:, :-1]), axis=1)
        
        if self.return_sequences:
            Y = [tf.cumsum(M[..., 0, :], reverse=self.reverse, axis=1)]
        else:
            Y = [tf.reduce_sum(M[..., 0, :], axis=1)]
        
        if not self.recursive_tensors:
            k = 1
            for m in range(1, self.num_levels):
                R = np.asarray([M[..., k, :]])
                k += 1
                for i in range(1, m+1):
                    d = min(i+1, self.order)
                    R_next = np.empty((d), dtype=tf.Tensor)
                    R_next[0] = M[..., k, :] *  tf.cumsum(tf.add_n(R.tolist()), reverse=self.reverse, exclusive=True, axis=1)
                    for j in range(1, d):
                        R_next[j] = 1 / tf.cast(j+1, dtype=X.dtype) * M[..., k, :] * R[j-1]
                    k += 1
                    R = R_next
                if self.return_sequences:
                    Y.append(tf.cumsum(tf.add_n(R.tolist()), reverse=self.reverse, axis=1))
                else:
                    Y.append(tf.reduce_sum(tf.add_n(R.tolist()), axis=1))
        else:
            R = np.asarray([M[..., 0, :]])
            for m in range(1, self.num_levels):
                d = min(m+1, self.order)
                R_next = np.empty((d), dtype=tf.Tensor)
                R_next[0] = M[..., m, :] * tf.cumsum(tf.add_n(R.tolist()), exclusive=True, reverse=self.reverse, axis=1)
                for j in range(1, d):
                    R_next[j] = 1 / tf.cast(j+1, dtype=X.dtype) * M[..., m, :] * R[j-1]
                R = R_next
                if self.return_sequences:
                    Y.append(tf.cumsum(tf.add_n(R.tolist()), reverse=self.reverse, axis=1))
                else:
                    Y.append(tf.reduce_sum(tf.add_n(R.tolist()), axis=1))
        if self.return_levels:
            return tf.stack(Y, axis=-2)
        else:
            return tf.add_n(Y)

    def compute_output_shape(self, input_shape):
        if self.return_sequences:
            if self.return_levels:
                return (input_shape[0], self.len_examples, self.num_levels, self.units)
            else:
                return (input_shape[0], self.len_examples, self.units)
        else:
            if self.return_levels:
                return (input_shape[0], self.num_levels, self.units)
            else:
                return (input_shape[0], self.units)
            