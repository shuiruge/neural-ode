r"""C.f. `./test_mnist_dense.py`"""

import numpy as np
import tensorflow as tf
from node.core import get_node_function
from node.solvers import RK4Solver
from node.utils.initializers import GlorotUniform


# for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


@tf.function
def normalize(x, axis=None):
    M = tf.reduce_max(x, axis, keepdims=True)
    m = tf.reduce_min(x, axis, keepdims=True)
    return (x - m) / (M - m + 1e-8)


class MyLayer(tf.keras.layers.Layer):

    def __init__(self, units, dt, num_grids, **kwargs):
        super().__init__(**kwargs)
        self.dt = dt
        self.num_grids = num_grids

        t0 = tf.constant(0.)
        self.tN = t0 + num_grids * dt

        self._model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu',
                                  kernel_initializer=GlorotUniform(1e-1)),
            tf.keras.layers.Dense(units, activation='relu',
                                  kernel_initializer=GlorotUniform(1e-1)),
        ])
        self._model.build([None, units])

        @tf.function
        def fn(t, x):
            z = self._model(x)
            with tf.GradientTape() as g:
                g.watch(x)
                r = normalize(x, axis=-1)
            return g.gradient(r, x, z)

        self._node_fn = get_node_function(
            RK4Solver(self.dt), tf.constant(0.), fn)

    def call(self, x):
        y = self._node_fn(self.tN, x)
        return y


def process(X, y):
    X = X / 255.
    X = tf.reshape(X, [-1, 28 * 28])
    y = tf.one_hot(y, 10)
    return tf.cast(X, tf.float32), tf.cast(y, tf.float32)


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = process(x_train, y_train)
x_test, y_test = process(x_test, y_test)

model = tf.keras.Sequential([
    tf.keras.layers.Input([28 * 28]),
    tf.keras.layers.Dense(64, activation='relu'),
    MyLayer(64, dt=1e-1, num_grids=10),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train,
          epochs=10,
          validation_data=(x_test, y_test))
