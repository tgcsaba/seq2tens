import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Layer, BatchNormalization, Dense, Conv1D, GlobalAveragePooling1D, Add, Reshape, Multiply, Lambda

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

class TSFlatten(Layer):
    
    def __init__(self, **kwargs):
        """
        Keras Layer for flattening the state-space of a batch of sequences. Works analogously to flatten,
        but preserves the time dimension (first axis after the batch).
        """
        super().__init__(**kwargs)
        self.supports_masking = True
    
    def build(self, input_shape):
        assert len(input_shape) >= 3        
        super().build(input_shape)
    
    def call(self, inputs, mask=None):
        dyn_shape = tf.shape(inputs)
        outputs = tf.reshape(inputs, (dyn_shape[0], dyn_shape[1], tf.reduce_prod(dyn_shape[2:])))
        outputs.set_shape(self.compute_output_shape(inputs.shape))
        return outputs
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], np.prod(input_shape[2:]))

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

class TSReshape(Layer):
    def __init__(self, input_shape, **kwargs):
        """
        Keras Layer for reshaping a batch of sequences. Works analogously to reshape, but preserves the time dimension (first axis after the batch),
        allowing to omit the time-dimension from input_shape (i.e. no need to waste the useful -1 wildcard on the time axis).
        """
        super().__init__(**kwargs)
        self.input_shape = tuple(input_shape)
        self.supports_masking = True
    
    def build(self, input_shape):
        assert len(input_shape) >= 3        
        super().build(input_shape)
    
    def call(self, X, mask=None):
        return tf.reshape(X, (tf.shape(X)[0], tf.shape(X)[1]) + self.input_shape)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1]) + self.input_shape

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
        
class SqueezeAndExcite(Layer):
    def __init__(self, channels, squeeze_factor, **kwargs):
        """
        Keras Layer implementing squeeze-and-excitation blocks to use with convolutional layers, see: https://arxiv.org/abs/1709.01507.

        Args:
            channels (int): The number of channels in the incoming batch of sequences of shape (batch_size, len_sequences, channels),
                            needs to be specified since the internal projection layers must be initialized in the constructor (rather than when build is called).
            squeeze_factor (int): The factor by which the features dimension of the squeezed block is reduced
                                  (i.e. it will have channels // squeeze_factor number of dimensions in the features axis)  
        """        
    
        Layer.__init__(self, **kwargs)
        
        self.pool = GlobalAveragePooling1D()
        self.broadcast = Reshape((1, -1))
        self.squeeze = Dense(channels // squeeze_factor, activation='relu', use_bias=False)
        self.expand = Dense(channels, activation='sigmoid', use_bias=False)
        self.mult = Multiply()
        
        self.channels = channels
        self.squeeze_factor = squeeze_factor
    
    def call(self, inputs):
        block = self.broadcast(self.expand(self.squeeze(self.pool(inputs))))
        outputs = self.mult([inputs, block])
        return outputs
        
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
        
class MergeShortcut(Layer):
    def __init__(self, channels, normalize_shortcut=True, normalize_residual=True, return_sequences=True, **kwargs):
        """
        Keras Layer for merging the shortcut branch into the residual branch, see: https://arxiv.org/abs/1512.03385. 
        When called, it must take a list or tuple of len=2, the first of which is the shortcut branch and the second the residual branch.

        Args:
            channels (int): the number of channels in the incoming residual branch, of shape i.e. (..., channels) 
                            needs to be specified since the shortcut projection must be initialized in the constructor (rather than when build is called)
            normalize_shortcut (bool, optional): Whether to apply batch norm to the shortcut branch,
                                                 set this to False if e.g. it has been normalized a-priori to calling this layer. Defaults to True.
            normalize_residual (bool, optional): Whether to apply batch norm to the residual branch, 
                                                 set this to False if e.g. it has been normalized a-priori to calling this layer. Defaults to True.
            return_sequences (bool, optional): Whether the incoming residual branch is a batch of sequences (i.e. of shape (batch_size, len_sequences, channels))
                                               or simply a batch of features (i.e. (batch_size, channels)). Defaults to True.
        """
             
        Layer.__init__(self, **kwargs)
        if return_sequences:
            self.proj_sc = Conv1D(channels, 1)
        else:
            self.pool_sc = GlobalAveragePooling1D()
            self.proj_sc = Dense(channels)
        if normalize_shortcut:
            self.bn_sc = BatchNormalization()
        if normalize_residual:
            self.bn_res = BatchNormalization()
        
        self.add = Add()
        self.rescale = Lambda(lambda x: x / tf.sqrt(2.))
            
        self.channels = channels
        self.normalize_shortcut = normalize_shortcut
        self.normalize_residual = normalize_residual
        self.return_sequences = return_sequences
        
    def call(self, inputs):
        shortcut = inputs[0]
        residual = inputs[1]
        if not self.return_sequences:
            shortcut = self.pool_sc(shortcut)
        shortcut = self.proj_sc(shortcut)
        if self.normalize_shortcut:
            shortcut = self.bn_sc(shortcut)
        if self.normalize_residual:
            residual = self.bn_res(residual)
        merged = self.add([shortcut, residual])    
        merged = self.rescale(merged)
        return merged
        
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------