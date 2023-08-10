import abc
from warnings import warn

import time
import numpy as np

from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout

from sklearn.utils.class_weight import compute_sample_weight

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from ..utils import EarlyStoppingAlwaysRestore

from ._utils import _check_for_hparam_in_dict, _normalize_list

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
    
class ModelBase(Model, metaclass=abc.ABCMeta):
    def __init__(self, batch_size=0.1, num_epochs=2000, optimizer='adam', callbacks='auto', dropout=None, max_batch_size=16, min_batch_size=4, opt_hparams=None,
                 cb_hparams=None, **kwargs):
        """
        Base model class, any classes inheriting from this must override the build_network and default_loss_name methods; optionally the default_metric_names method.

        Args:
            batch_size (float, optional): batch size to use during training. If < 1., then interpreted as a percentage of the training date. Defaults to 0.1.
            num_epochs (int, optional): max number of epochs to train for. Defaults to 2000.
            optimizer (str, optional): optimizer for training. Defaults to 'adam'.
            callbacks (str, optional): a list of callbacks to use, if 'auto' then uses EarlyStopping and ReduceLROnPlateau. Defaults to 'auto'.
            dropout (float, optional): Dropout to apply to the final feature layer, if None then no dropout is used. Defaults to None.
            max_batch_size (int, optional): Max usable batch size, when the batch_size argument is specified as a ratio of the training data. Defaults to 16.
            min_batch_size (int, optional): Min usable batch size, when the batch_size argument is specified as a ratio of the training data. Defaults to 4.
            opt_hparams (dict, optional): A dict-like object containing any hparams that are then passed on to the optimizer. Defaults to None.
            cb_hparams (dict, optional):  A dict-like object containing any hparams that are then passed on to the call backs in case callbacks=='auto'. Defaults to None. 
                                          Possible entries are: 'monitor', 'es_patience', 'lr_patience', 'lr_factor', 'lr_min'
        Kwargs:
            _All_ additional keyword arguments are passed along to the build_network function, and hence, should contain the hyperparameters of the network.
                                         

        Raises:
            ValueError: if opt_hparams is not a dict-like object
            ValueError: if cb_hparams is not a dict-like object
        """
        Model.__init__(self)
        
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.callbacks = callbacks
        self.dropout = dropout
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        
        if opt_hparams is not None and not isinstance(opt_hparams, dict):
            raise ValueError('ValueError | ModelBase: if specified, opt_hparams must be a dict-like object.')
        self.opt_hparams = opt_hparams
        
        if cb_hparams is not None and not isinstance(cb_hparams, dict):
            raise ValueError('ValueError | ModelBase: if specified, cb_hparams must be a dict-like object.')
        self.cb_hparams = cb_hparams    
        
        self.hparams = kwargs
        
        self.network = self.build_network(**kwargs)
        
        if self.dropout is not None:
            self.dp_layer = Dropout(dropout)
            
    @abc.abstractmethod
    def build_network(self):
        pass
    
    @property
    @abc.abstractmethod
    def default_loss_name(self):
        pass
    
    @property
    def default_metric_names(self):
        pass
    
    def call(self, inputs):
        features = self.network(inputs)
        if self.dropout is not None:
            features = self.dp_layer(features)
        return features            
    
    def compile(self, loss=None, metrics=None, verbose=True):
        optimizer = keras.optimizers.get(self.optimizer)
        if self.opt_hparams is not None:
            for hp_name, hp_value in self.opt_hparams.items():
                setattr(optimizer, hp_name, hp_value)
        
        if loss is None:
            loss = keras.losses.get(self.default_loss_name)
        elif verbose:
            print(f'Information | ModelBase.compile: loss has been overridden from \'{self.default_loss_name}\' to new loss.')
        loss = keras.losses.get(loss)
                                
        default_metrics = _normalize_list(self.default_metric_names)
        metrics = _normalize_list(metrics)
        if metrics is not None:
            if default_metrics is not None:
                metrics = default_metrics + metrics
                if verbose:
                    print('Information | ModelBase.compile: Default model metrics have been merged with new metrics.')
            elif verbose:
                print('Information | ModelBase.compile: New metrics have been initialized. ')
        else:
            metrics = default_metrics
        metrics = [keras.metrics.get(m) for m in metrics]
        
        Model.compile(self, optimizer=optimizer, loss=loss, metrics=metrics)
        
    def get_default_batch_size(self, num_train):
        if self.batch_size is None:
            batch_size = num_train
        elif self.batch_size < 1.:
            # specified as ratio of training data
            batch_size =  int(np.floor(num_train * self.batch_size))
        else:
            batch_size = self.batch_size 
        if self.min_batch_size is not None:
            batch_size = max(batch_size, self.min_batch_size)
        if self.max_batch_size is not None:
            batch_size = min(batch_size, self.max_batch_size)
        return batch_size


    def get_default_callbacks(self, verbose=True):
        monitor = _check_for_hparam_in_dict(self.cb_hparams, 'monitor', 'loss')
        mode = _check_for_hparam_in_dict(self.cb_hparams, 'mode', 'auto')
        es_patience = _check_for_hparam_in_dict(self.cb_hparams, 'es_patience', 500)
        lr_patience = _check_for_hparam_in_dict(self.cb_hparams, 'lr_patience', 100)
        lr_factor = _check_for_hparam_in_dict(self.cb_hparams, 'lr_factor', 1./2)
        lr_min = _check_for_hparam_in_dict(self.cb_hparams, 'lr_min', 1e-4)
        
        reduce_lr = ReduceLROnPlateau(monitor=monitor, mode=mode, patience=lr_patience, factor=lr_factor, min_lr=lr_min, min_delta=0., verbose=verbose)
        es = EarlyStoppingAlwaysRestore(monitor=monitor, mode=mode, patience=es_patience, restore_best_weights=True, min_delta=0., verbose=verbose)
        return [reduce_lr, es]
        
    def _check_for_callbacks_merge(self, callbacks, default_callbacks, verbose=True):
        callbacks = _normalize_list(callbacks)
        default_callbacks = _normalize_list(default_callbacks)
        if callbacks is not None: 
            if default_callbacks is not None:
                callbacks = callbacks + default_callbacks
                if verbose:
                    print('Information | ModelBase: Callbacks passed at initialization have been merged with new callbacks passed at fitting time. ',
                          'Please make sure no duplicates have been passed.')  
            elif verbose:
                    print('Information | ModelBase: Callbacks have been passed during fitting time and are now in use.')
            return callbacks
        else:
            return default_callbacks
                 
        
    def fit(self, X_train, y_train, loss=None, metrics=None, verbose=True, **kwargs):
    
        self.compile(loss=loss, metrics=metrics)
        
        num_train = X_train.shape[0]
        default_batch_size = self.get_default_batch_size(num_train)
        batch_size = _check_for_hparam_in_dict(kwargs, 'batch_size', default_batch_size)
        if batch_size != default_batch_size:
            if verbose:
                print(f'Information | ModelBase.fit: \'batch_size\' has been overridden from \'{default_batch_size}\' to \'{batch_size}\'')
                
        default_num_epochs = self.num_epochs
        num_epochs = _check_for_hparam_in_dict(kwargs, 'epochs', default_num_epochs)
        num_epochs = _check_for_hparam_in_dict(kwargs, 'num_epochs', num_epochs)
        if num_epochs != default_num_epochs:
            if verbose:
                print(f'Information | ModelBase.fit: \'num_epochs\' has been overridden from \'{default_num_epochs}\' to \'{num_epochs}\'')
        
        callbacks = self.get_default_callbacks(verbose=verbose) if self.callbacks == 'auto' else self.callbacks
        if 'callbacks' in kwargs:
            callbacks = self._check_for_callbacks_merge(kwargs['callbacks'], callbacks, verbose=verbose)
            del kwargs['callbacks']
        callbacks = _normalize_list(callbacks)
            
        history = Model.fit(self, X_train, y_train, batch_size=batch_size, epochs=num_epochs, callbacks=callbacks, verbose=verbose, **kwargs)
        return history
        
    def fit_generator(self, generator, loss=None, metrics=None, verbose=True, **kwargs):
    
        self.compile(loss=loss, metrics=metrics)

        print(f'Information | ModelBase.fit_generator: \'batch_size\' hyperparameter is ignored.')
                
        default_num_epochs = self.num_epochs
        num_epochs = _check_for_hparam_in_dict(kwargs, 'epochs', default_num_epochs)
        num_epochs = _check_for_hparam_in_dict(kwargs, 'num_epochs', num_epochs)
        if num_epochs != default_num_epochs:
            if verbose:
                print(f'Information | ModelBase.fit_generator: \'num_epochs\' has been overridden from \'{default_num_epochs}\' to \'{num_epochs}\'')
        
        callbacks = self.get_default_callbacks(verbose=verbose) if self.callbacks == 'auto' else self.callbacks
        if 'callbacks' in kwargs:
            callbacks = self._check_for_callbacks_merge(kwargs['callbacks'], callbacks)
            del kwargs['callbacks']
        callbacks = _normalize_list(callbacks)
            
        history = Model.fit(self, generator, epochs=num_epochs, callbacks=callbacks, verbose=verbose, **kwargs)
        return history
        
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------