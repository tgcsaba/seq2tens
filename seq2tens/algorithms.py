import numpy as np
import tensorflow as tf

from warnings import warn

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

def _normalize_weights_mode(mode):
    mode = mode.lower().replace('_', '')[:3]
    return mode

def low_rank_seq2tens(sequences, order, kernel, bias=None, embedding_order=1, mode='ind', reverse=False, mask=None, return_sequences=False):
    """
    Tensorflow implementation of the Low-rank Seq2Tens (LS2T) map
    
    Args:
        sequences (array): a batch of sequences of shape (batch_size, len_sequences, num_features)
        order (int): the order of the LS2T layer (truncation level for the free algebra)
        kernel (array): the components of the rank-1 weight tensors, of shape:
                         - (order*(order+1) // 2, num_features, num_functionals) if mode=='ind'
                         - (order, num_features, num_functionals) if mode=='rec'
                         - (1, num_features, num_functionals) if mode=='sym'
        bias (array, optional): an optional bias term for each rank-1 weight component, of shape
                                - (order*(order+1) // 2, num_functionals) if mode=='ind'
                                - (order, num_functionals) if mode=='rec'
                                - (1, num_functionals) if mode=='sym'
                                Defaults to None.
        embedding_order (int, optional): embedding order for the lift into the free algebra, must be between [1, order]. Defaults to 1.
        mode (str, optional): Specifies which variant of the LS2T is used, possible values are : ['ind', 'rec', 'sym']. Defaults to 'ind'.
        reverse (bool, optional): Whether to compute the LS2T in reverse mode, only matters if return_sequences==True
                                  (then fixes the end-point instead of the starting point of the expanding windows). Defaults to False.
        mask (array, optional): an optional mask array of shape (batch_size, len_sequences). Defaults to None.
        return_sequences (bool, optional): Whether to use the LS2T as a seq2seq transform. Defaults to False.

    Returns:
        features (array): resulting features for the given batch of shape (batch_size, len_sequences, num_functionals, order) if return_sequences==True
                          else (batch_size, num_functionals, order)
    """

    mode = _normalize_weights_mode(mode)

    batch_size, len_sequences, num_features = tf.unstack(tf.shape(sequences))
    
    num_components = order * (order+1) // 2 if mode=='ind' else order if mode=='rec' else 1
    
    kernel = tf.reshape(kernel, [num_components, num_features, -1])
    
    projections = tf.matmul(tf.reshape(sequences, [1, -1, num_features]), kernel)
    projections = tf.reshape(projections, [num_components, batch_size, len_sequences, -1])
    
    if bias is not None:
        projections += bias[:, None, None, :]
    
    if mask is not None:
        projections *= tf.cast(mask[None, :, :, None], projections.dtype)
    
    features = _low_rank_seq2tens(projections,  order, embedding_order=embedding_order, mode=mode, reverse=reverse, return_sequences=return_sequences)
    return features


def _low_rank_seq2tens(projections, order, embedding_order=1, mode='ind', reverse=False, return_sequences=False):
    
    if embedding_order == 1:
        if mode == 'sym':
            return _low_rank_seq2tens_first_order_embedding_symmetric_weights(projections, order, reverse=reverse, return_sequences=return_sequences)
        elif mode == 'rec':
            return _low_rank_seq2tens_first_order_embedding_recursive_weights(projections, order, reverse=reverse, return_sequences=return_sequences)
        elif mode == 'ind':
            return _low_rank_seq2tens_first_order_embedding_indep_weights(projections, order, reverse=reverse, return_sequences=return_sequences)
    else:
        if mode == 'sym':
            return _low_rank_seq2tens_higher_order_embedding_symmetric_weights(projections, order, embedding_order, reverse=reverse, return_sequences=return_sequences)
        elif mode == 'rec':
            return _low_rank_seq2tens_higher_order_embedding_recursive_weights(projections, order, embedding_order, reverse=reverse, return_sequences=return_sequences)
        elif mode == 'ind':
            return _low_rank_seq2tens_higher_order_embedding_indep_weights(projections, order, embedding_order, reverse=reverse, return_sequences=return_sequences)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Implementations of the variants of the LS2T algorithms
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

def _low_rank_seq2tens_first_order_embedding_symmetric_weights(M, order, reverse=False, return_sequences=False):
    
    if return_sequences:
        Y = [tf.cumsum(M, reverse=reverse, axis=1)]
    else:
        Y = [tf.reduce_sum(M, axis=1)]

    R = M
    for m in range(1, order):
        R = M * tf.cumsum(R, exclusive=True, reverse=reverse, axis=1)
        
        if return_sequences:
            Y.append(tf.cumsum(R, reverse=reverse, axis=1))
        else:
            Y.append(tf.reduce_sum(R, axis=1))

    return tf.stack(Y, axis=-1)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

def _low_rank_seq2tens_first_order_embedding_recursive_weights(M, order, reverse=False, return_sequences=False):
    
    if return_sequences:
        Y = [tf.cumsum(M[0], reverse=reverse, axis=1)]
    else:
        Y = [tf.reduce_sum(M[0], axis=1)]

    R = M[0]
    for m in range(1, order):
        R = M[m] * tf.cumsum(R, exclusive=True, reverse=reverse, axis=1)
        
        if return_sequences:
            Y.append(tf.cumsum(R, reverse=reverse, axis=1))
        else:
            Y.append(tf.reduce_sum(R, axis=1))

    return tf.stack(Y, axis=-1)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

def _low_rank_seq2tens_first_order_embedding_indep_weights(M, order, reverse=False, return_sequences=False):
    
    if return_sequences:
        Y = [tf.cumsum(M[0], reverse=reverse, axis=1)]
    else:
        Y = [tf.reduce_sum(M[0], axis=1)]

    k = 1
    for m in range(1, order):
        R = M[k]
        k += 1
        for i in range(1, m+1):
            R = M[k] *  tf.cumsum(R, exclusive=True, reverse=reverse, axis=1)
            k += 1
        if return_sequences:
            Y.append(tf.cumsum(R, reverse=reverse, axis=1))
        else:
            Y.append(tf.reduce_sum(R, axis=1))
    
    return tf.stack(Y, axis=-1)
    
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------   

def _low_rank_seq2tens_higher_order_embedding_symmetric_weights(M, order, embedding_order, reverse=False, return_sequences=False):
    
    if return_sequences:
        Y = [tf.cumsum(M, reverse=reverse, axis=1)]
    else:
        Y = [tf.reduce_sum(M, axis=1)]

    R = np.empty((1,), dtype=type(M))
    R[0] = M
    R_sum = M
    for m in range(1, order):
        d = min(m+1, embedding_order)
        R_next = np.empty((d), dtype=type(M))
        R_next[0] = M * tf.cumsum(R_sum, exclusive=True, reverse=reverse, axis=1)
        for j in range(1, d):
            R_next[j] = 1./tf.cast(j+1, M.dtype) * M * R[j-1]
        R = R_next
        R_sum = tf.add_n(R.tolist())
        if return_sequences:
            Y.append(tf.cumsum(R_sum, reverse=reverse, axis=1))
        else:
            Y.append(tf.reduce_sum(R_sum, axis=1))

    return tf.stack(Y, axis=-1)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

def _low_rank_seq2tens_higher_order_embedding_recursive_weights(M, order, embedding_order, reverse=False, return_sequences=False):
    
    if return_sequences:
        Y = [tf.cumsum(M[0], reverse=reverse, axis=1)]
    else:
        Y = [tf.reduce_sum(M[0], axis=1)]

    R = np.empty((1,), dtype=type(M))
    R[0] = M[0]
    R_sum = M[0]
    for m in range(1, order):
        d = min(m+1, embedding_order)
        R_next = np.empty((d), dtype=type(M))
        R_next[0] = M[m] * tf.cumsum(R_sum, exclusive=True, reverse=reverse, axis=1)
        for j in range(1, d):
            R_next[j] = 1./tf.cast(j+1, M.dtype) * M[m] * R[j-1]
        R = R_next
        R_sum = tf.add_n(R.tolist())
        if return_sequences:
            Y.append(tf.cumsum(R_sum, reverse=reverse, axis=1))
        else:
            Y.append(tf.reduce_sum(R_sum, axis=1))

    return tf.stack(Y, axis=-1)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

def _low_rank_seq2tens_higher_order_embedding_indep_weights(M, order, embedding_order, reverse=False, return_sequences=False):
    
    if return_sequences:
        Y = [tf.cumsum(M[0], reverse=reverse, axis=1)]
    else:
        Y = [tf.reduce_sum(M[0], axis=1)]

    k = 1
    for m in range(1, order):
        R = np.empty((1,), dtype=type(M))
        R[0] = M[k]
        R_sum = M[k]
        k += 1
        for i in range(1, m+1):
            d = min(i+1, embedding_order)
            R_next = np.empty((d), dtype=type(M))
            R_next[0] = M[k] * tf.cumsum(R_sum, exclusive=True, reverse=reverse, axis=1)
            for j in range(1, d):
                R_next[j] = 1./tf.cast(j+1, M.dtype) * M[k] * R[j-1]
            k += 1
            R = R_next
            R_sum = tf.add_n(R.tolist())
        if return_sequences:
            Y.append(tf.cumsum(R_sum, reverse=reverse, axis=1))
        else:
            Y.append(tf.reduce_sum(R_sum, axis=1))

    return tf.stack(Y, axis=-1)
    
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------