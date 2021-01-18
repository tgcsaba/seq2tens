import numpy as np
import tensorflow as tf

from tensorflow.keras import initializers

class LS2TInitializer(initializers.Initializer):
    def __init__(self, num_levels, recursive_tensors):
        self.num_levels = num_levels
        self.recursive_tensors = recursive_tensors

    def __call__(self, shape, dtype=None):
        _, num_features, units = shape
        if self.recursive_tensors:
            limit = tf.sqrt(3.) * tf.sqrt(2. / (num_features + units))
            kernels = [tf.random.uniform((num_features, units), minval=-limit, maxval=limit, dtype=dtype)]
            for m in range(2, self.num_levels+1):
                limit = tf.sqrt(3.) * tf.sqrt((tf.pow(float(num_features), float(m-1)) + units) / (tf.pow(float(num_features), float(m)) + units))
                kernels.append(tf.random.uniform((num_features, units), minval=-limit, maxval=limit, dtype=dtype))
            kernel = tf.stack(kernels, axis=0)
        else:
            kernels = []
            for m in range(1, self.num_levels+1):
                limit = tf.sqrt(3.) * tf.pow(2. / (tf.pow(float(num_features), float(m)) + units), 1. / (2*float(m)))
                kernels.append(tf.random.uniform((m, num_features, units), minval=-limit, maxval=limit, dtype=dtype))
            kernel = tf.concat(kernels, axis=0)
            
        return kernel

    def get_config(self): 
        return {'num_levels': self.num_levels, 'recursive_tensors': self.recursive_tensors}