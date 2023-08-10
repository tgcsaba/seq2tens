import numpy as np
import tensorflow as tf

from math import sqrt

from tensorflow.keras.initializers import Initializer

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

def _compute_target_variances(num_features, num_functionals, order, recursive_weights=False):
    """
    Computes the target variances for initializing the components of the rank-1 tensor components according to Glorot init.
    See the paper Appendix for details.

    Args:
        num_features (int): the dimension of the last axis for the input sequences, i.e. the state-space dimension
        num_functionals (int): the number of distinct linear functionals for each level of the free algebra
        order (int): the order of the LS2T layer (truncation level for the free algebra)
        recursive_weights (bool, optional): Whether to use the recursive formulation of the LS2T. Defaults to False.

    Returns:
        variances (list): a list of len==order
    """
    
    
    if recursive_weights:
        variances = [2. / (num_features + num_functionals)] \
                    + [(tf.pow(float(num_features), float(i)) + num_functionals) / (tf.pow(float(num_features), float(i+1)) + num_functionals)
                       for i in range(1, order)]
    else:
        variances = [tf.pow(2. / (tf.pow(float(num_features), float(i)) + num_functionals), 1. / (float(i)))
                     for i in range(1, order+1)]
    return variances
        
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

class LS2TUniformInitializer(Initializer):
    def __init__(self, order, recursive_weights=False):
        """
        Uniform Glorot initializer for LS2T.

        Args:
            order (int): the order of the LS2T layer (truncation level for the free algebra)
            recursive_weights (bool, optional): Whether to use the recursive formulation of the LS2T. Defaults to False.
        """
        self.order = order
        self.recursive_weights = recursive_weights

    def __call__(self, shape, dtype=None):
        _, num_features, num_functionals = shape
        variances = _compute_target_variances(num_features, num_functionals, self.order, self.recursive_weights)
        kernels = []
        for i, variance in enumerate(variances):
            current_shape = (1, num_features, num_functionals) if self.recursive_weights else (i+1, num_features, num_functionals)
            limit = tf.sqrt(3.) * tf.sqrt(variance)
            kernel = tf.random.uniform(current_shape, minval=-limit, maxval=limit, dtype=dtype)
            kernels.append(kernel)
        kernel = tf.concat(kernels, axis=0)
        return kernel

    def get_config(self): 
        return {'order': self.order, 'recursive_weights': self.recursive_weights}

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

class LS2TNormalInitializer(Initializer):
    def __init__(self, order, recursive_weights=False):
        """
        Normal Glorot initializer for LS2T.

        Args:
            order (int): the order of the LS2T layer (truncation level for the free algebra)
            recursive_weights (bool, optional): Whether to use the recursive formulation of the LS2T. Defaults to False.
        """
        self.order = order
        self.recursive_weights = recursive_weights
        
    def __call__(self, shape, dtype=None):
        _, num_features, num_functionals = shape
        variances = _compute_target_variances(num_features, num_functionals, self.order, self.recursive_weights)
        kernels = []
        for i, variance in enumerate(variances):
            current_shape = (1, num_features, num_functionals) if self.recursive_weights else (i+1, num_features, num_functionals)
            stddev = tf.sqrt(variance)
            kernel = tf.random.normal(current_shape, stddev=stddev, dtype=dtype)  
            kernels.append(kernel)
        kernel = tf.concat(kernels, axis=0)
        return kernel

    def get_config(self): 
        return {'order': self.order, 'recursive_weights': self.recursive_weights}
        
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------