import numpy as np
import tensorflow as tf

from ._initializers import LS2TUniformInitializer, LS2TNormalInitializer
from ._constraints import SigmoidConstraint
from ..algorithms import low_rank_seq2tens

from tensorflow.keras import initializers, regularizers, constraints

from tensorflow.keras.layers import Layer

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

class TimeCoord(Layer):
    def __init__(self, **kwargs):
        """
        Keras Layer, which takes batch of sequences of shape (batch_size, len_sequences, num_features)
        to a batch of sequences of shape (batch_size, len_sequences, num_features+1), where the extra coordinate
        is a centered time-equispaced variable in the range [-1, 1].
        """
        super().__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) == 3        
        super().build(input_shape)
        
    def call(self, inputs, mask=None):
        if mask is not None:            
            mask = tf.cast(mask, inputs.dtype)
            time = (tf.cumsum(mask, axis=1, exclusive=True) / (tf.reduce_sum(mask, keepdims=True, axis=1) - 1.) * 2. - mask)[..., None]
            outputs = tf.concat((time, inputs), axis=-1)
        else:
            num_examples = tf.shape(inputs)[0]
            time = tf.tile(tf.range(tf.cast(tf.shape(inputs)[1], inputs.dtype), dtype=inputs.dtype)[None, :, None], [num_examples, 1, 1])
            time *= 2. / (tf.cast(tf.shape(inputs)[1], inputs.dtype) - 1.)
            time -= 1.
            outputs = tf.concat((time, inputs), axis=-1)
        return outputs
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2] + 1)
        
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- 
        
class TSDifference(Layer):
    def __init__(self, **kwargs):
        """
        Keras Layer, which takes batch of sequences and preserves their shape. Concatenates an extra zero observation at the beginning,
        then differences the sequences along the time axis, such that:
        output[:, 0, :] = input[:, 0, :] 
        output[:, i, :] = input[:, i, :] - input[:, i-1, :] for i >= 1.
        """
        super().__init__(**kwargs)
        self.supports_masking = True
    
    def build(self, input_shape):
        assert len(input_shape) == 3        
        super().build(input_shape)
    
    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[[...] + (inputs.shape.ndims - mask.shape.ndims)*[None]] # broadcast shape
            mask = tf.cast(mask, inputs.dtype) # cast type
            inputs *= mask
        inputs = tf.concat((tf.zeros_like(inputs[:, :1]), inputs), axis=1)
        return inputs[:, 1:] - inputs[:, :-1]
    
    def compute_output_shape(self, input_shape):
        return input_shape

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
        
class LS2T(Layer):
    def __init__(self, num_functionals, order, embedding_order=1, recursive_weights=False, reverse=False, use_normalization=False, use_bias=True,
                 kernel_initializer='ls2t_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None, return_sequences=False, **kwargs):
        """
        Keras Layer wrapper of the LS2T map

        Args:
            num_functionals (int): The number of distinct linear functionals for each level of the free algebra.
            order (int): The order of the LS2T layer (i.e. truncation level for the free algebra).
            embedding_order (int, optional): Embedding order for the lift into the free algebra, must be between [1, order]. Defaults to 1.
            recursive_weights (bool, optional): Whether to use the recursive formulation of the LS2T. Defaults to False.
            reverse (bool, optional): Whether to compute the LS2T in reverse mode, only matters if return_sequences==True
                                      (then fixes the end-point instead of the starting point of the expanding windows). Defaults to False.
            use_bias (bool, optional): Whether to use an additional bias term for each linear functional on the state-space . Defaults to True.
            kernel_initializer (str, optional): Initializer for the vector components of the rank-1 tensors.
                                                Possible values are: 'ls2t_uniform', 'ls2t_normal'. Defaults to 'ls2t_uniform'.
            bias_initializer (str, optional): If use_bias==True, then the initializer for the biases. Defaults to 'zeros'.
            return_sequences (bool, optional): Whether to return sequences or . Defaults to False.
        """
    
    
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)
        
        self.num_functionals = num_functionals
        self.order = order
        self.embedding_order = embedding_order
        
        self.recursive_weights = recursive_weights
        self.use_bias = use_bias
        self.use_normalization = use_normalization
        self.reverse = reverse        
        self.return_sequences = return_sequences
        
        if kernel_initializer.lower().replace('_', '') == 'ls2tuniform':
            self.kernel_initializer = LS2TUniformInitializer(order, recursive_weights)
        elif kernel_initializer.lower().replace('_', '') == 'ls2tnormal':
            self.kernel_initializer = LS2TNormalInitializer(order, recursive_weights)
        else:
            self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
            
        if use_bias:
            self.bias_initializer = initializers.get(bias_initializer)
            self.bias_regularizer = regularizers.get(bias_regularizer)
            self.bias_constraint = constraints.get(bias_constraint)
        
        if use_normalization:
            self.normalization_initializer = initializers.get('zeros')
            self.normalization_constraint = SigmoidConstraint()
        
        self.supports_masking = True        

    def build(self, input_shape):
        self.num_features = input_shape.as_list()[-1]
        
        assert len(input_shape) == 3
        
        self.num_components = int(self.order * (self.order+1) / 2.) if not self.recursive_weights else self.order
        
        self.kernel = self.add_weight(shape=tf.TensorShape([self.num_components, self.num_features, self.num_functionals]), name='kernel',
                                      initializer=self.kernel_initializer, regularizer=self.kernel_regularizer, constraint=self.kernel_constraint,
                                      dtype=self.dtype, trainable=True)
        
        if self.use_bias:
            self.bias = self.add_weight(shape=tf.TensorShape([self.num_components, self.num_functionals]), name='bias', initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer, constraint=self.bias_constraint, dtype=self.dtype, trainable=True)
            
        if self.use_normalization:
            self.normalization = self.add_weight(shape=tf.TensorShape([self.order]), name='normalization', initializer=self.normalization_initializer,
                                                 constraint=self.normalization_constraint, dtype=self.dtype, trainable=True)
        
        super().build(input_shape)
        
    def call(self, inputs, mask=None):
        
        outputs = low_rank_seq2tens(inputs, self.order, self.kernel, bias=self.bias if self.use_bias else None, embedding_order=self.embedding_order,
                                    mode='rec' if self.recursive_weights else 'ind', reverse=self.reverse, mask=mask, return_sequences=self.return_sequences)
        
        if self.use_normalization:
            if self.return_sequences:
                n = tf.range(1, tf.shape(inputs)[1]+1, dtype=self.dtype)
                k = tf.range(1, self.order+1, dtype=self.dtype)
                # n choose k
                count = tf.math.lgamma(n[:, None]+1) - tf.math.lgamma(k[None, :]+1) \
                        - tf.math.lgamma(tf.maximum(n[:, None]-k[None, :]+1, 1.))
                count *= self.normalization[None, :]
                count = tf.exp(count)
                outputs /= count[None, :, None, :]
            else:
                n = tf.cast(tf.shape(inputs)[1], self.dtype)
                k = tf.range(1, self.order+1, dtype=self.dtype)
                # n choose k
                count = tf.exp(tf.math.lgamma(n+1) - tf.math.lgamma(k+1) - tf.math.lgamma(n-k+1))
                outputs /= count**self.normalization
        
        return outputs
            
    def compute_output_shape(self, input_shape):
        if self.return_sequences:
            return (input_shape[0], input_shape[1], self.num_functionals, self.order)
        else:
            return (input_shape[0], self.num_functionals, self.order)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------