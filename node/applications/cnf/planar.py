"""
References:
1. https://github.com/kmkolasinski/deep-learning-notes/blob/master/seminars/2019-03-Neural-Ordinary-Differential-Equations/cnf.py  # noqa:E501
"""

from node.applications.cnf.base import CNF


class Planar(CNF):
    """C.f., ref(1)."""
    def __init__(self, hyper_net, t0, t1, **kwargs):
        super().__init__(t0, t1, **kwargs)
        self.hyper_net = hyper_net

    @tf.function
    def _dynamics(self, t, x):
        W, b, u = self.hyper_net(t)
        n_esamble = self.hyper_net.n_ensemble
        x = tf.tile(tf.expand_dims(x, 0), [n_esamble, 1, 1])
        h = tf.tanh(tf.matmul(x, W) + b)
        return tf.reduce_mean(tf.matmul(h, u), 0)

    @tf.function
    def _log_prob_dynamics(self, t, x_and_log_prob):
        x, _ = x_and_log_prob
        W, b, u = self.hyper_net(t)
        n_esamble = self.hyper_net.n_ensemble
        x = tf.tile(tf.expand_dims(x, 0), [n_esamble, 1, 1])

        with tf.GradientTape() as g:
            g.watch(x)
            h = tf.tanh(tf.matmul(x, W) + b)
            reduced_h = tf.reduce_sum(h)
        grad_h = g.gradient(reduced_h, x)
        grad_log_prob = tf.squeeze(
            tf.reduce_mean(
                -tf.matmul(grad_h, tf.transpose(u, [0, 2, 1])),
                axis=0))
        grad_x = tf.reduce_mean(tf.matmul(h, u), 0)
        return [grad_x, grad_log_prob]


class HyperNet(tf.keras.Model):
    """Auxillary class for `Planar`. C.f., ref(1)."""

    def __init__(self, input_dim, hidden_dim, n_ensemble, **kwargs):
        super().__init__(**kwargs)

        blocksize = n_ensemble * input_dim
        self._layers = [
            tf.keras.layers.Dense(hidden_dim, activation='tanh'),
            tf.keras.layers.Dense(hidden_dim, activation='tanh'),
            tf.keras.layers.Dense(3 * blocksize + n_ensemble),
        ]
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_ensemble = n_ensemble
        self.blocksize = blocksize

    def call(self, t):
        t = tf.reshape(t, [1, 1])
        params = t
        for layer in self._layers:
            params = layer(params)

        # restructure
        params = tf.reshape(params, [-1])
        W = tf.reshape(
            params[:self.blocksize],
            shape=[self.n_ensemble, self.input_dim, 1])

        U = tf.reshape(
            params[self.blocksize:2 * self.blocksize],
            shape=[self.n_ensemble, 1, self.input_dim])

        G = tf.sigmoid(
            tf.reshape(
                params[2 * self.blocksize:3 * self.blocksize],
                shape=[self.n_ensemble, 1, self.input_dim]
            )
        )
        U = U * G
        B = tf.reshape(params[3 * self.blocksize:], [self.n_ensemble, 1, 1])
        return [W, B, U]

    def compute_output_shape(self, input_shape):
        W_shape = [self.n_ensemble, self.input_dim, 1]
        B_shape = [self.n_ensemble, 1, 1]
        U_shape = [self.n_ensemble, 1, self.input_dim]
        return W_shape, B_shape, U_shape
