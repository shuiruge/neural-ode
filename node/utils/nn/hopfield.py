import tensorflow as tf
from node.core import get_node_function


@tf.function
def normalize(x, axis=None):
    M = tf.reduce_max(x, axis, keepdims=True)
    m = tf.reduce_min(x, axis, keepdims=True)
    return (x - m) / (M - m + 1e-8)


class HopfieldLayer(tf.keras.layers.Layer):
    """Hopfield layer = convolution + normalization"""

    def __init__(self, filters, kernel_size, solver, t, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.solver = solver
        self.t = t

        self.convolve = tf.keras.layers.Conv2D(
            filters, kernel_size, activation='relu', padding='same')

        @tf.function
        def pvf(t, x):
            z = self.convolve(x)
            with tf.GradientTape() as g:
                g.watch(x)
                r = normalize(x, axis=[-3, -2, -1])
            return g.gradient(r, x, z)

        self._pvf = pvf
        self._node_fn = get_node_function(solver, 0., pvf)

    def call(self, x):
        y = self._node_fn(self.t, x)
        return y
