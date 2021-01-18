import numpy as np
import tensorflow as tf

from .layers import LS2T

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv1D, Dense

def LSUVReinitializer(model, X_init, margin=0.1, max_iter=10, jitter=1e-6):
    """
    https://arxiv.org/abs/1511.06422
    @article{mishkin2015all,
    title={All you need is a good init},
    author={Mishkin, Dmytro and Matas, Jiri},
    journal={arXiv preprint arXiv:1511.06422},
    year={2015}
    """
    margin = 0.1
    max_iter = 10
    jitter = 1e-6
    for i, layer in enumerate(model.layers[:-1]):
        print('{}: {}'.format(i, model.layers[i]))
        if isinstance(layer, LS2T):
            weights = layer.get_weights()
            weights[0] = np.stack([np.linalg.qr(A)[0] for A in weights[0]], axis=0)
            layer.set_weights(weights)

            temp_model = Model(model.input, layer.output)

            values = temp_model.predict(X_init)
            if layer.return_sequences:
                values = values.reshape([-1, layer.num_levels, layer.units])
            stds = np.std(values, axis=0)
            
            if layer.recursive_tensors:
                stds = stds**(1./(layer.num_levels))
                rescales = np.concatenate((np.ones((1, layer.units)), np.cumprod(stds[:-1], axis=0)), axis=0)
                rescales = np.maximum(rescales, jitter)
                rescales = stds / rescales
                rescales = rescales**(float(layer.num_levels))
            else:
                rescales = np.concatenate([np.tile(stds[m:m+1, :]**(1./(m+1)), [m+1, 1]) for m in range(layer.num_levels)], axis=0)
            rescales = np.maximum(rescales, jitter)
            weights[0] /= rescales[:, None, :]
            layer.set_weights(weights)
        elif isinstance(layer, Conv1D):
            weights = layer.get_weights()
            weights[0] = np.linalg.qr(weights[0].reshape([-1, layer.filters]))[0].reshape([layer.kernel_size[0], -1, layer.filters])
            layer.set_weights(weights)

            temp_model = Model(model.input, layer.output)
            values = temp_model.predict(X_init)
            values = values.reshape([-1, layer.filters])
            stds = np.std(values, axis=0)
            current_iter = 0
            while np.any(np.abs(stds - 1.0) > margin) and current_iter < 10:
                rescales = np.maximum(stds, jitter)
                weights[0] /= rescales[None, None, :]
                layer.set_weights(weights)
                values = temp_model.predict(X_init)
                values = values.reshape([-1, layer.filters])
                stds = np.std(values, axis=0)
                current_iter += 1
        elif isinstance(layer, Dense):
            weights = layer.get_weights()
            weights[0] = np.linalg.qr(weights[0])[0]
            layer.set_weights(weights)

            temp_model = Model(model.input, layer.output)
            values = temp_model.predict(X_init)
            values = values.reshape([-1, layer.units])
            stds = np.std(values, axis=0)
            current_iter = 0
            while np.any(np.abs(stds - 1.0) > margin) and current_iter < 10:
                stds = np.maximum(stds, jitter)
                weights[0] /= stds[None, :]
                layer.set_weights(weights)
                values = temp_model.predict(X_init)
                values = values.reshape([-1, layer.units])
                stds = np.std(values, axis=0)
                current_iter += 1
    return model