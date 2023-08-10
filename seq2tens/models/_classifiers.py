import abc

from warnings import warn

import numpy as np

from tensorflow import keras
from tensorflow.keras.layers import Dense

from sklearn.utils.class_weight import compute_sample_weight

from ._base import ModelBase
from ..layers import DeepLS2TNet, FCN, FCNLS2TNet, ResNet

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

class ClassifierBase(ModelBase, metaclass=abc.ABCMeta):
    def __init__(self, num_classes, mode='multi_sparse', **kwargs):
        """
        Base classifier class inheriting from ModelBase with (sparse_)categorical_crossentropy loss and accuracy as metric by default, additionally overriding
        a lot of the class member methods for convenience in classifications.
        The NetworkToClassifier function can be used to wrap a network (without a final classification layer) as a classifier by 
            FCNClassifier = NetworkToClassifier(FCN)
        where FCN can be a class inheriting from tf.keras.Layers.Layer, implementing the network architecture according to the keras Layer paradigm,
        see e.g. https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer,
        or it can simply be a function returning a Layer object.

        Args:
            num_classes (int): the number of classes in the classification task
            mode (str, optional): Takes three possible values [\'binary\', \'multi_sparse\', \'multi_onehot\'];
                                  - \'binary\' : if num_classes==2 uses sigmoid class layer with binary_crossentropy loss; else falls back to \'multi_sparse\'.
                                  - \'multi_sparse\' : uses softmax class layer with sparse_categorical_crossentropy loss
                                  - \'multi_onehot\' : uses softmax class layer with categorical_crossentropy loss; requires labels in one-hot format.
        """
        
        ModelBase.__init__(self, **kwargs)
        
        self.num_classes = num_classes
        mode = mode.lower()
        if mode not in ['binary', 'multi_sparse', 'multi_onehot']:
            warn(f'Warning | ClassifierBase.__init__: Unknown classifier mode: \'{mode}\', falling back to \'multi_sparse\'.')
        elif num_classes != 2 and mode == 'binary':
            mode = 'multi_sparse'
            warn(f'Warning | ClassifierBase.__init__: \'binary\' classifier mode specified, but num_classes != 2, falling back to \'multi_sparse\'.')
        self.mode = mode
        self.class_layer = self._build_class_layer(num_classes, mode)
    
    def _build_class_layer(self, num_classes, mode):
        if num_classes==2 and mode=='binary':
            return Dense(1, activation='sigmoid')
        else:
            return Dense(num_classes, activation='softmax')    
        
    def call(self, inputs):
        features = ModelBase.call(self, inputs)
        outputs = self.class_layer(features)
        return outputs
    
    def fit(self, X_train, y_train, balance_loss=False, **kwargs):
        if ('sample_weight' in kwargs and kwargs['sample_weight'] is not None) or ('class_weight' in kwargs and kwargs['class_weight'] is not None):
            warn('Warning | ClassifierBase.fit: \'sample_weight\' or \'class_weight\' specified among keyword arguments, ignoring balance_loss kwarg.')
        elif balance_loss:
            kwargs['sample_weight'] = compute_sample_weight('balanced', y_train)
        return ModelBase.fit(self, X_train, y_train, **kwargs)
    
    def fit_generator(self, generator, balance_loss=False, **kwargs):
        if balance_loss:
            warn('Warning | ClassifierBase.fit_generator: balance_loss kwargs is only ignored for fit_generator; class_weights must be specified externally.')
        return ModelBase.fit_generator(self, generator, **kwargs)
    
    def evaluate(self, x=None, y=None, balance_loss=False, **kwargs):
        if ('sample_weight' in kwargs and kwargs['sample_weight'] is not None) or ('class_weight' in kwargs and kwargs['class_weight'] is not None):
            warn('Warning | ClassifierBase.evaluate: \'sample_weight\' or \'class_weight\' specified among keyword arguments, ignoring balance_loss kwarg.')
        elif balance_loss:
            kwargs['sample_weight'] = compute_sample_weight('balanced', y)
        return ModelBase.evaluate(self, x=x, y=y, **kwargs)
            
    @property
    def default_loss_name(self):
        if self.num_classes == 2 and self.mode == 'binary':
            return 'binary_crossentropy'
        elif self.mode == 'multi_onehot':
            return 'categorical_crossentropy'
        else:    
            return 'sparse_categorical_crossentropy'
    
    @property
    def default_metric_names(self):
        if self.num_classes==2 and self.mode=='binary':
            return 'binary_accuracy'
        elif self.mode == 'multi_sparse':
            return 'sparse_categorical_accuracy'
        else:
            return 'accuracy'

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# wrap networks as classifiers
            
def NetworkToClassifier(LayerBlock):
    class ClassifierClassFromLayerBlock(ClassifierBase):
        def build_network(self, **kwargs):
            return LayerBlock(**kwargs)
    return ClassifierClassFromLayerBlock
    
DeepLS2TClassifier = NetworkToClassifier(DeepLS2TNet)
FCNClassifier = NetworkToClassifier(FCN)
FCNLS2TClassifier = NetworkToClassifier(FCNLS2TNet)
ResNetClassifier = NetworkToClassifier(ResNet)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------