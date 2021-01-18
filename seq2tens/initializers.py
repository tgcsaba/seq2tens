import numpy as np
import tensorflow as tf

from tensorflow.keras.initializers import Initializer

def _compute_target_variances(num_features, units, num_levels, recursive_weights=False):
    """
    Computes the target variances for initializing the components of the rank-1 weight tensors according to Glorot init
    """
    if recursive_weights:
        variances = [2. / (num_features + units)] + [(tf.pow(float(num_features), float(i)) + units) / (tf.pow(float(num_features), float(i+1)) + units) for i in range(1, num_levels)]
    else:
        variances = [tf.pow(2. / (tf.pow(float(num_features), float(i)) + units), 1. / (float(i))) for i in range(1, num_levels+1)]
    return variances
        

class LS2TUniformInitializer(Initializer):
    """
    Uniform Glorot initializer for LS2T
    """
    def __init__(self, num_levels, recursive_weights=False):
        self.num_levels = num_levels
        self.recursive_weights = recursive_weights

    def __call__(self, shape, dtype=None):
        _, num_features, units = shape
        variances = _compute_target_variances(num_features, units, self.num_levels, self.recursive_weights)
        kernels = []
        for i, variance in enumerate(variances):
            limit = tf.sqrt(3.) * tf.sqrt(variance)
            current_shape = (1, num_features, units) if self.recursive_weights else (i+1, num_features, units)
            kernels.append(tf.random.uniform(current_shape, minval=-limit, maxval=limit, dtype=dtype))
        kernel = tf.concat(kernels, axis=0)
        return kernel

    def get_config(self): 
        return {'num_levels': self.num_levels, 'recursive_tensors': self.recursive_tensors}

class LS2TNormalInitializer(Initializer):
    """
    Normal Glorot initializer for LS2T
    """
    def __init__(self, num_levels, recursive_weights=False):
        self.num_levels = num_levels
        self.recursive_weights = recursive_weights

    def __call__(self, shape, dtype=None):
        _, num_features, units = shape
        variances = _compute_target_variances(num_features, units, self.num_levels, self.recursive_weights)
        kernels = []
        for i, variance in enumerate(variances):
            stdev = tf.sqrt(variance)
            current_shape = (1, num_features, units) if self.recursive_weights else (i+1, num_features, units)
            kernels.append(tf.random.normal((num_features, units), stdev=stdev, dtype=dtype))
        kernel = tf.concat(kernels, axis=0)
        return kernel

    def get_config(self): 
        return {'num_levels': self.num_levels, 'recursive_tensors': self.recursive_tensors}