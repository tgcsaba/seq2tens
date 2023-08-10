import numpy as np
import tensorflow as tf

from warnings import warn

from tensorflow import keras

from tensorflow.keras.layers import Layer, Input, Activation, BatchNormalization, Flatten, Dense, Conv1D, GlobalAveragePooling1D, Add, Reshape, Multiply, Lambda, Concatenate, Dropout, SpatialDropout1D

from ._core import TimeCoord, TSDifference, LS2T
from ._utils import TSFlatten, MergeShortcut, SqueezeAndExcite

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

class DeepLS2TNet(Layer):
    def __init__(self, num_functionals=64, order=2, embedding_order=1, depth=3, recursive_weights=True, difference=True, use_normalization=False, use_bias=True,
                 batch_norm=True, renorm=False, spatial_dropout=None, return_sequences=False, **kwargs):
        
        Layer.__init__(self, **kwargs)
        
        if not batch_norm and renorm:
            warn('Warning | DeepLS2TNet: renorm option is ignored when batch-norm is False.')
        
        if difference:
            self.diffs = [TSDifference() for i in range(depth)]
        
        self.ls2ts = [LS2T(num_functionals, order, embedding_order=embedding_order, recursive_weights=recursive_weights, use_normalization=use_normalization,
                           use_bias=use_bias, return_sequences=return_sequences if i==depth-1 else True) for i in range(depth)]
        if batch_norm:
            self.bns = [BatchNormalization(scale=False, center=False, renorm=renorm) for i in range(depth)]
            
        self.flattens = [Flatten() if i==depth-1 and not return_sequences else TSFlatten() for i in range(depth)]
        
        if spatial_dropout is not None:
            self.dps = [Dropout(spatial_dropout) if i==depth-1 and not return_sequences else SpatialDropout1D(spatial_dropout) for i in range(depth)]
        
        self.num_functionals = num_functionals
        self.order = order
        self.embedding_order = embedding_order
        self.depth = depth
        self.recursive_weights = recursive_weights 
        self.difference = difference 
        self.use_normalization = use_normalization
        self.use_bias = use_bias
        self.batch_norm = batch_norm
        self.renorm = renorm
        self.spatial_dropout = spatial_dropout
        self.return_sequences = return_sequences 
            
    def call(self, inputs):
        layer = inputs
        for i in range(self.depth):
            if self.difference:
                layer = self.diffs[i](layer)
            layer = self.ls2ts[i](layer)
            if self.batch_norm:
                layer = self.bns[i](layer)
            layer = self.flattens[i](layer)
            if self.spatial_dropout is not None:
                layer = self.dps[i](layer)
        return layer

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

class FCN(Layer):
    def __init__(self, filters=128, depth=3, kernels=[8,5,3], time_coord=True, batch_norm=True, renorm=False, residual=False, squeeze_and_excite=False, 
                 squeeze_factor=None, wider_inner_layers=True, spatial_dropout=None, return_sequences=False, **kwargs):
    
        Layer.__init__(self, **kwargs)
        
        if np.isscalar(kernels):
            kernels = [kernels for i in range(depth)]
        elif not isinstance(kernels, list) and not isinstance(kernels, tuple):
            raise ValueError('ValueError | FCN: kernels should either be a scalar or an array of kernel sizes.')
        else:
            kernels = list(kernels)
            if len(kernels) != depth:
                raise ValueError('ValueError | FCN: if kernels is a list-like container, then must have len(kernels)==depth.')
        
        if np.isscalar(filters):
            filters = [filters if i==0 or i==depth-1 or not wider_inner_layers else 2*filters for i in range(depth)]
        elif not isinstance(filters, list) and not isinstance(filters, tuple):
            raise ValueError('ValueError | FCN: filters should either be a scalar or an array of kernel sizes.')
        else:
            filters = list(filters)
            if len(kernels) != depth:
                raise ValueError('ValueError | FCN: if filters is a list-like container, then must have len(filters)==depth.')
            
            
        if not batch_norm and renorm:
            warn('Warning | FCN: renorm option is ignored when batch-norm is False.')
            
        if squeeze_and_excite and squeeze_factor is None:
            raise ValueError('ValueError | FCN: when squeeze_and_excite=True, squeeze_factor must be set to a positive int (e.g. 8 or 16).')
        elif not squeeze_and_excite and squeeze_factor is not None:
            warn(f'Warning | FCN: when squeeze_and_excite=False the value of squeeze_factor=\'{squeeze_factor}\' is ignored.')
            
        if residual:
            if not batch_norm:
                warn('Warning | FCN: Using the residual option without batch_norm is not expected to work well, '
                     'since then you most likely need specific initialization for the layers (e.g. FixUp init).')
            self.merge_sc = MergeShortcut(filters[-1], normalize_shortcut=batch_norm, normalize_residual=False, return_sequences=True)
            
        if time_coord:
            self.times = [TimeCoord() for i in range(depth)]
        
        self.convs = [Conv1D(filters[i], kernels[i], padding='same') for i in range(depth)]
        
        if batch_norm:
            self.bns = [BatchNormalization(renorm=renorm) for i in range(depth)]
            
        self.acts = [Activation('relu') for i in range(depth)]
        
        if squeeze_and_excite:
            self.squeeze_blocks = [SqueezeAndExcite(filters[i], filters[i] / squeeze_factor) for i in range(depth-1)]
            
        if spatial_dropout is not None:
            self.dps = [SpatialDropout1D(spatial_dropout) for i in range(depth)]
        
        if not return_sequences:
            self.pool = GlobalAveragePooling1D()
                             
        self.filters = filters
        self.depth = depth
        self.kernels = kernels
        self.time_coord = time_coord
        self.batch_norm = batch_norm
        self.renorm = renorm
        self.residual = residual
        self.squeeze_and_excite = squeeze_and_excite
        self.squeeze_factor = squeeze_factor
        self.wider_inner_layers = wider_inner_layers
        self.spatial_dropout = spatial_dropout
        self.return_sequences = return_sequences
    
    def call(self, inputs):
        layer = inputs
        if self.residual:
            shortcut = inputs
        for i in range(self.depth):
            if self.time_coord:
                layer = self.times[i](layer)
            layer = self.convs[i](layer)            
            if self.batch_norm:
                layer = self.bns[i](layer)
            if self.residual and i == self.depth-1:
                layer = self.merge_sc([shortcut, layer])
            layer = self.acts[i](layer)
            if self.squeeze_and_excite and i < self.depth-1:
                layer = self.squeeze_blocks[i](layer)
            if self.spatial_dropout is not None:
                layer = self.dps[i](layer)
        if not self.return_sequences:
            layer = self.pool(layer)
        return layer
        
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

class FCNLS2TNet(Layer):
    def __init__(self, fcn_filters=64, fcn_depth=3, fcn_kernels=[8,5,3], fcn_squeeze_and_excite=False, fcn_squeeze_factor=None, ls2t_functionals=None,
                 ls2t_order=2, ls2t_embedding_order=1, ls2t_depth=3, ls2t_recursive=True, ls2t_difference=True, ls2t_normalization=False, time_coord=True,
                 batch_norm=True, renorm=False, residual_fcn=True, residual_ls2t=True, spatial_dropout=None, return_sequences=False, **kwargs):
        
        Layer.__init__(self, **kwargs)
        
        ls2t_functionals = ls2t_functionals or fcn_filters
        
        if (residual_fcn or residual_ls2t) and not batch_norm:
            warn('Warning | FCNLS2TNet: Using the residual_fcn or residual_ls2t options without batch_norm is not expected to work well, '
                 'since then you most likely need specific initialization for the layers (e.g. FixUp init).')
            
        
        if residual_fcn:
            self.merge_inp = MergeShortcut(fcn_filters, normalize_shortcut=batch_norm, normalize_residual=batch_norm, return_sequences=True)
            
        if residual_ls2t:
            self.merge_fcn = MergeShortcut(ls2t_functionals * ls2t_order, normalize_shortcut=batch_norm, normalize_residual=False,
                                           return_sequences=return_sequences)
                
        
        self.fcn_block = FCN(fcn_filters, depth=fcn_depth, kernels=fcn_kernels, time_coord=time_coord, 
                             batch_norm=batch_norm, renorm=renorm, residual=False, squeeze_and_excite=fcn_squeeze_and_excite,
                             squeeze_factor=fcn_squeeze_factor, wider_inner_layers=True, spatial_dropout=spatial_dropout, return_sequences=True)
        
        self.deep_ls2t_block = DeepLS2TNet(ls2t_functionals, order=ls2t_order, embedding_order=ls2t_embedding_order, depth=ls2t_depth,
                                           recursive_weights=ls2t_recursive, difference=ls2t_difference, use_normalization=ls2t_normalization, use_bias=time_coord,
                                           batch_norm=batch_norm, renorm=renorm, spatial_dropout=spatial_dropout, return_sequences=return_sequences)
            
        self.fcn_filters = fcn_filters
        self.fcn_depth = fcn_depth
        self.fcn_kernels = fcn_kernels
        self.ls2t_functionals = ls2t_functionals 
        self.ls2t_order = ls2t_order
        self.ls2t_embedding_order = ls2t_embedding_order
        self.ls2t_depth = ls2t_depth
        self.ls2t_recursive = ls2t_recursive
        self.ls2t_difference = ls2t_difference
        self.ls2t_normalization = ls2t_normalization
        self.time_coord = time_coord
        self.batch_norm = batch_norm
        self.renorm = renorm
        self.residual_fcn = residual_fcn
        self.residual_ls2t = residual_ls2t
        self.spatial_dropout = spatial_dropout
        self.return_sequences = return_sequences
    
    def call(self, inputs):
        fcn = self.fcn_block(inputs)
        if self.residual_ls2t:
            shortcut_fcn = fcn
        if self.residual_fcn:
            fcn = self.merge_inp([inputs, fcn])
        ls2t = self.deep_ls2t_block(fcn)
        if self.residual_ls2t:
            ls2t = self.merge_fcn([shortcut_fcn, ls2t])
        return ls2t
        
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
        
class ResNet(Layer):
    def __init__(self, fcn_filters=64, fcn_depth=3, fcn_kernels=[8,5,3], time_coord=False, batch_norm=True, renorm=False, residual_blocks=3,
                 spatial_dropout=None, return_sequences=False, **kwargs):
        
        Layer.__init__(self, **kwargs)
        
        self.fcn_blocks = [FCN(fcn_filters if i==0 else 2*fcn_filters, depth=fcn_depth, kernels=fcn_kernels, time_coord=time_coord, batch_norm=batch_norm,
                               renorm=renorm, residual=True, wider_inner_layers=False, spatial_dropout=spatial_dropout,
                               return_sequences=True if i<residual_blocks-1 else return_sequences) for i in range(residual_blocks)]
        
        self.fcn_filters = fcn_filters
        self.fcn_depth = fcn_depth
        self.fcn_kernels = fcn_kernels
        self.time_coord = time_coord
        self.batch_norm = batch_norm
        self.renorm = renorm
        self.residual_blocks = residual_blocks
        self.spatial_dropout = spatial_dropout
        self.return_sequences = return_sequences
    
    def call(self, inputs):
        layer = inputs
        for i in range(self.residual_blocks):
            layer = self.fcn_blocks[i](layer)
        return layer

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------