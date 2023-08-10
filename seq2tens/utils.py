import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import Callback, EarlyStopping      

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

class ValidationMetrics(Callback):
    def __init__(self, metrics, X_val, y_val, batch_size=None, verbose=False):
        self.metrics = metrics
        self.X_val = X_val
        self.y_val = y_val
        self.batch_size = batch_size
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        y_pred_val = self.model.predict(self.X_val, batch_size=self.batch_size)
        
        for metric_name, metric in self.metrics.items():
            metric_value = metric(self.y_val, y_pred_val)
            val_metric_name = f'val_{metric_name}'
            logs[val_metric_name] = metric_value
            if self.verbose:
                print(f'\r{val_metric_name} : {metric_value:0.04f} - ', end='')

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

class EarlyStoppingAlwaysRestore(EarlyStopping):
    """
    A small upgrade on the standard EarlyStopping callback of keras. Fixes the issue that early stopping does not restore the best weights with the
    restore_best_weights option when the optimization ends by reaching the max number of epochs (rather than early stopping triggering the end of training).
    See:
        https://github.com/keras-team/keras/issues/12511
        https://github.com/tensorflow/tensorflow/issues/35634
    """  
    def on_train_end(self, logs=None):
        EarlyStopping.on_train_end(self, logs=logs)
        if self.restore_best_weights:
            if self.verbose > 0:
                print('Restoring model weights from the end of the best epoch.')
            self.model.set_weights(self.best_weights)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

class SacredLogger(Callback):
    """
    A logger callback for Sacred (https://github.com/IDSIA/sacred/), that logs the quantities monitored by the Model.fit function during training.
    """
    def __init__(self, _run):
        """
        Initializes a logger callback for Sacred that takes as argument a Sacred Run object.
        Args:
            _run (Run): A Sacred Run object that represents and manages a single run of an experiment.
        """
        self._run = _run
    def on_epoch_end(self, epoch, logs={}):
        for metric, value in logs.items():
            metric = metric.lower().replace('sparse_', '').replace('categorical_', '').replace('binary_', '')
            if metric[:4] == 'val_':
                metric = metric[4:]
                metric = f'validation.{metric}'
            else:
                metric = f'training.{metric}'
            self._run.log_scalar(metric, value, epoch)
            
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

def make_balanced_generator(X, y): 
    
    num_classes = np.unique(y).size
    num_ex_per_class = np.bincount(y)
    
    ds = tf.data.Dataset.from_tensor_slices((X, y))

    def _is_from_class(class_id, *args):
        if len(args) == 2:
            data, label = args
        if len(args) == 3:
            data, label, sample_weights = args
        return tf.math.equal(label, class_id)

    ds_class = [ds.filter(lambda *args: _is_from_class(c, *args)).shuffle(num_ex_per_class[c], reshuffle_each_iteration=True).repeat() for c in range(num_classes)]
    
    ds_balanced = tf.data.experimental.choose_from_datasets(ds_class, tf.data.Dataset.range(num_classes).repeat())
    
    return ds_balanced
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------