import numpy as np
import tensorflow as tf

def low_rank_seq2tens(sequences, kernel, num_levels, embedding_order=1, recursive_weights=False, bias=None, reverse=False, return_sequences=False, mask=None):
    """
    Tensorflow implementation of the Low-rank Seq2Tens (LS2T) map
    --------------------------------------------------
    Args
    ----
    :sequences: - a tensor of sequences of shape (num_examples, len_examples, num_features)
    :kernel: - a tensor of component vectors of rank-1 weight tensors of shape (num_components, num_features, num_functionals)
    :num_levels: - an int scalar denoting the cutoff degree in the features themselves (must be consistent with the 'num_components' dimension of 'kernel')
    :embedding_order: - an int scalar denoting the cutoff degree in the algebraic embedding
    :recursive_weights: - whether the rank-1 weight twensors are contructed in a recursive way (must be consistent with the shape of 'kernel')
    :bias: - a tensor of biases of shape (num_components, num_functionals)
    :reverse: - only changes the results with 'return_sequences=True', determines whether the output sequences are constructed by moving the starting point or ending point of subsequences
    """
    
    num_sequences, len_sequences, num_features = tf.unstack(tf.shape(sequences))

    num_components = int(num_levels * (num_levels+1) / 2.) if not recursive_weights else num_levels
    
    num_functionals = tf.shape(kernel)[-1]
        
    M = tf.matmul(tf.reshape(sequences, [1, -1, num_features]), kernel)
        
    M = tf.reshape(M, [num_components, num_sequences, len_sequences, num_functionals])
    
    if bias is not None:
        M += bias[:, None, None, :]
    
    if mask is not None:
        M = tf.where(mask[None, :, :, None], M, tf.zeros_like(M))

    if embedding_order == 1:
        if recursive_weights:
            return _low_rank_seq2tens_first_order_embedding_recursive_weights(M, num_levels, reverse=reverse, return_sequences=return_sequences)
        else:
            return _low_rank_seq2tens_first_order_embedding_indep_weights(M, num_levels, reverse=reverse, return_sequences=return_sequences)
    else:
        if recursive_weights:
            return _low_rank_seq2tens_higher_order_embedding_recursive_weights(M, num_levels, embedding_order, reverse=reverse, return_sequences=return_sequences)
        else:
            return _low_rank_seq2tens_higher_order_embedding_indep_weights(M, num_levels, embedding_order, reverse=reverse, return_sequences=return_sequences)


def _low_rank_seq2tens_first_order_embedding_recursive_weights(M, num_levels, reverse=False, return_sequences=False):
    
    if return_sequences:
        Y = [tf.cumsum(M[0], reverse=reverse, axis=1)]
    else:
        Y = [tf.reduce_sum(M[0], axis=1)]

    R = M[0]
    for m in range(1, num_levels):
        R = M[m] * tf.cumsum(R, exclusive=True, reverse=reverse, axis=1)
        
        if return_sequences:
            Y.append(tf.cumsum(R, reverse=reverse, axis=1))
        else:
            Y.append(tf.reduce_sum(R, axis=1))

    return tf.stack(Y, axis=-2)
    
def _low_rank_seq2tens_first_order_embedding_indep_weights(M, num_levels, reverse=False, return_sequences=False):
    
    if return_sequences:
        Y = [tf.cumsum(M[0], reverse=reverse, axis=1)]
    else:
        Y = [tf.reduce_sum(M[0], axis=1)]

    k = 1
    for m in range(1, num_levels):
        R = M[k]
        k += 1
        for i in range(1, m+1):
            R = M[k] *  tf.cumsum(R, exclusive=True, reverse=reverse, axis=1)
            k += 1
        if return_sequences:
            Y.append(tf.cumsum(R, reverse=reverse, axis=1))
        else:
            Y.append(tf.reduce_sum(R, axis=1))
    
    return tf.stack(Y, axis=-2)
        

def _low_rank_seq2tens_higher_order_embedding_recursive_weights(M, num_levels, embedding_order, reverse=False, return_sequences=False):
    
    if return_sequences:
        Y = [tf.cumsum(M[0], reverse=reverse, axis=1)]
    else:
        Y = [tf.reduce_sum(M[0], axis=1)]

    R = np.empty((1,), dtype=type(M))
    R[0] = M[0]
    R_sum = M[0]
    for m in range(1, num_levels):
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

    return tf.stack(Y, axis=-2)

def _low_rank_seq2tens_higher_order_embedding_indep_weights(M, num_levels, embedding_order, reverse=False, return_sequences=False):
    
    if return_sequences:
        Y = [tf.cumsum(M[0], reverse=reverse, axis=1)]
    else:
        Y = [tf.reduce_sum(M[0], axis=1)]

    k = 0
    for m in range(1, num_levels):
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

    return tf.stack(Y, axis=-2)