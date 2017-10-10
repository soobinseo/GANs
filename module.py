import tensorflow as tf
import numpy as np
import hyperparams as hp

def lrelu(x, leak=0.2):

    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    output = f1 * x + f2 * abs(x)
    return output


def fully_connected(input, output_shape, norm_fn=tf.contrib.layers.batch_norm, initializer=tf.truncated_normal_initializer(stddev=0.02), scope="fc", is_last=False):
    with tf.variable_scope(scope):
        input_shape = input.get_shape()[-1].value
        W = tf.get_variable("weight", [input_shape, output_shape], initializer=initializer)
        b = tf.get_variable("bias", [output_shape], initializer=initializer)
        fc = tf.add(tf.matmul(input, W), b)

        if not is_last:
            if norm_fn is not None:
                fc = norm_fn(fc)
            output = lrelu(fc)
        else:
            output = fc

    return output

def conv2d(tensor,
           output_dim,
           filter_height=hp.filter_height,
           filter_width=hp.filter_width,
           stride=hp.stride,
           activation_fn=tf.nn.relu,
           norm_fn=tf.contrib.layers.batch_norm,
           initializer=tf.truncated_normal_initializer(stddev=0.02),
           scope="name",
           reflect=False,
           padding="SAME"):

    with tf.variable_scope(scope):
        if reflect:
            tensor = tf.pad(tensor, [[0, 0], [1, 1], [1, 1], [0, 0]])
            tensor_shape = tensor.get_shape().as_list()
            filter = tf.get_variable('filter', [filter_height, filter_width, tensor_shape[-1], output_dim], initializer=initializer)
            conv = tf.nn.conv2d(tensor, filter, strides=[1, stride, stride, 1], padding='VALID')
            if norm_fn is None:
                bn = conv
            else:
                bn = tf.contrib.layers.batch_norm(conv)

            # bn = tf.nn.dropout(bn, keep_prob=keep_prob)
            if activation_fn is not None:
                output = activation_fn(bn)
            else:
                output = bn

            return output
        else:
            tensor_shape = tensor.get_shape().as_list()
            filter = tf.get_variable('filters', [filter_height, filter_width, tensor_shape[-1], output_dim], initializer=initializer)
            conv = tf.nn.conv2d(tensor, filter, strides=[1, stride, stride, 1], padding=padding)

            if norm_fn is None:
                bn = conv
            else:
                bn = tf.contrib.layers.batch_norm(conv)

            if activation_fn is not None:
                output = activation_fn(bn)
            else:
                output = bn

            return output

# source from Keras library

def reduce_var(x, axis=None, keepdims=False):
    """Variance of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the variance.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the variance of elements of `x`.
    """
    m = tf.reduce_mean(x, axis=axis, keep_dims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keep_dims=keepdims)

def reduce_std(x, axis=None, keepdims=False):
    """Standard deviation of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the standard deviation.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the standard deviation of elements of `x`.
    """
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))