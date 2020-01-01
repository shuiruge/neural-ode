import tensorflow as tf


@tf.function
def normalize(x, axis=None):
    M = tf.reduce_max(x, axis=axis, keepdims=True)
    m = tf.reduce_min(x, axis=axis, keepdims=True)
    return (x - m) / (M - m)


if __name__ == '__main__':

    x = tf.constant([[1, 2, 3]], dtype=tf.float32)

    with tf.GradientTape() as g:
        g.watch(x)
        y = normalize(x, axis=1)
        tf.print(y)

    print(g.gradient(y, x, x + 1.))
