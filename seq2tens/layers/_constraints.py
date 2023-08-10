import numpy as np
import tensorflow as tf

from math import sqrt

from tensorflow.keras.constraints import Constraint
from tensorflow.keras.activations import sigmoid

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

class SigmoidConstraint(Constraint):
    def __init__(self, high=1.0, low=0.0):
        self.high = high
        self.low = low

    def __call__(self, W):
        return sigmoid(W) * (self.high - self.low) + self.low

    def get_config(self):
        return {'high': self.high,
                'low': self.low}
        
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------