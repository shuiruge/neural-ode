import tensorflow as tf


@tf.custom_gradient
def soft_reduce_max(x, axis=None, keepdims=False, factor=1.):
    score = tf.nn.softmax(factor * x, axis=None)
    y = tf.reduce_sum(x * score, axis=axis, keepdims=True)

    @tf.function
    def grad_fn(grad_y):
        return (score * grad_y) * (1. + factor * (x - y))

    if keepdims:
        return y, grad_fn
    else:
        return tf.squeeze(y, axis=axis), grad_fn


def soft_reduce_min(x, axis=None, keepdims=False, factor=1.):

    def reduce_max(x):
        return soft_reduce_max(x, axis=axis, keepdims=keepdims, factor=factor)

    return -reduce_max(-x)


# XXX: Currently, `tf.function` does not support keywords arguments
# in graph mode. C.f.,
# https://github.com/tensorflow/tensorflow/blob/5fc194159e3b0cf644a33fdd2df21549d3c6e973/tensorflow/python/ops/custom_gradient.py#L301  # noqa:E501
def soft_normalize(x, axis=None, factor=1.):
    M = soft_reduce_max(x, axis=axis, keepdims=True, factor=factor)
    m = soft_reduce_min(x, axis=axis, keepdims=True, factor=factor)
    return (x - m) / (M - m)


def normalize(x, axis=None):
    """Hard version"""
    M = tf.reduce_max(x, axis=axis, keepdims=True)
    m = tf.reduce_min(x, axis=axis, keepdims=True)
    return (x - m) / (M - m)


if __name__ == '__main__':

    x = tf.constant([[1, 2, 3]], dtype=tf.float32)

    with tf.GradientTape() as g:
        g.watch(x)
        y = soft_normalize(x, axis=1, factor=10.)
        tf.print(y)

    print(g.gradient(y, x, x + 1.))
