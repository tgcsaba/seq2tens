import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import keras
from keras import backend as K
from keras import initializers, regularizers, layers, activations, constraints
from keras.layers import RNN, Layer, InputSpec

from tensorflow.python.keras.layers.recurrent import DropoutRNNCellMixin

class RecurrentLRS(RNN):

    def __init__(self, units, num_levels, num_features, decoupled=False, activation='tanh', use_bias=False, kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None,
                 bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
                 dropout=0., recurrent_dropout=0., return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False,
                 **kwargs):

        cell = RecurrentLRSCell(units, num_levels, num_features, decoupled=decoupled, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer,
                                recurrent_initializer=recurrent_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
                                recurrent_regularizer=recurrent_regularizer, bias_regularizer=bias_regularizer, kernel_constraint=kernel_constraint,
                                recurrent_constraint=recurrent_constraint, bias_constraint=bias_constraint, dropout=dropout, recurrent_dropout=recurrent_dropout, 
                                dtype=kwargs.get('dtype'))
        
        super(RecurrentLRS, self).__init__(cell, return_sequences=return_sequences, return_state=return_state, go_backwards=go_backwards, stateful=stateful, unroll=unroll, **kwargs)
        
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.input_spec = [InputSpec(ndim=3)]

    def call(self, inputs, mask=None, training=None, initial_state=None):
        self.cell.reset_dropout_mask()
        self.cell.reset_recurrent_dropout_mask()
        return super(RecurrentLRS, self).call(inputs, mask=mask, training=training, initial_state=initial_state)

    @property
    def units(self):
        return self.cell.units
    
    @property
    def num_levels(self):
        return self.cell.num_levels
    
    @property
    def num_features(self):
        return self.cell.num_features
    
    @property
    def decoupled(self):
        return self.cell.decoupled

    @property
    def activation(self):
        return self.cell.activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    def get_config(self):
        config = {
            'units': self.units,
            'num_levels': self.num_levels,
            'num_features': self.num_features,
            'decoupled' : self.decoupled,
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'dropout': self.dropout,
            'recurrent_dropout': self.recurrent_dropout
        }
        base_config = super(RecurrentLRS, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class RecurrentLRSCell(DropoutRNNCellMixin, Layer):

    def __init__(self, units, num_levels, num_features, decoupled=False, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0., recurrent_dropout=0., **kwargs):             
        
        self.units = units
        self.num_features = num_features
        self.num_levels = num_levels
        self.activation = activations.get(activation)
        self.decoupled = decoupled
        self.use_bias = use_bias
        
        if self.decoupled:
            self.state_size = [self.num_features, self.num_levels * self.units, self.num_levels * self.units, self.num_levels * self.units]
        else:
            self.state_size = [self.num_features, self.num_levels * self.units, self.num_levels * self.units]
            
        self.output_size = self.num_levels * self.units
        
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        
        super(RecurrentLRSCell, self).__init__(**kwargs)

    def build(self, input_shape):
        assert self.num_features == input_shape[-1]
        
        self.kernel = self.add_weight(shape=(self.num_features,  self.num_levels * self.units), initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer, constraint=self.kernel_constraint, name='kernel')
        
        self.recurrent_kernel = self.add_weight(shape=(self.num_levels * self.units, self.num_levels * self.units), initializer=self.recurrent_initializer,
                                                regularizer=self.recurrent_regularizer, constraint=self.recurrent_constraint, name='recurrent_kernel')
        
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.num_levels * self.units,), initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer, constraint=self.bias_constraint, name='bias')
        else:
            self.bias = None
            
        self.built = True

    def call(self, inputs, states, training=None):
        num_samples = tf.shape(inputs)[0]
        xt = inputs
        
        xt_1 = states[0]
        ht_1 = states[1]
        ht_2 = states[2]
        
        if self.decoupled:
            st_1 = states[3]
        
        dxt_1 = xt - xt_1
        dht_2 = ht_1 - ht_2
        
        dp_mask = self.get_dropout_mask_for_cell(dxt_1, training)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(dht_2, training)
        
        if dp_mask is not None:
            dxt_1 *= dp_mask
        if rec_dp_mask is not None:
            dht_2 *= rec_dp_mask
            
        ut = tf.matmul(dxt_1, self.kernel) + tf.matmul(dht_2, self.recurrent_kernel)
        
        if self.use_bias:
            ut += self.bias[None, :]
        
        if self.decoupled:
            st = st_1 + tf.concat((tf.ones((num_samples, self.units), dtype=xt.dtype), st_1[:, :-self.units]), axis=1) * ut
            ht = st
        else:
            ht = ht_1 + tf.concat((tf.ones((num_samples, self.units), dtype=xt.dtype), ht_1[:, :-self.units]), axis=1) * ut
        
        if self.activation is not None:
            ht = self.activation(ht)
        
        if self.decoupled:
            return ht, [xt, ht, ht_1, st]
        else:
            return ht, [xt, ht, ht_1]
    
    def get_config(self):
        config = {
            'units': self.units,
            'num_features' : self.num_features,
            'num_levels' : self.num_levels,
            'decoupled' : self.decoupled,
            'activation' : activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'dropout': self.dropout,
            'recurrent_dropout': self.recurrent_dropout
        }
        base_config = super(RecurrentLRSCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))