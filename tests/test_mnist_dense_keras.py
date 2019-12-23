import numpy as np
import tensorflow as tf
from node.fix_grid import RKSolver
from node.wrapper import get_node_function


# for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class MyLayer(tf.keras.layers.Layer):

    def __init__(self, dt, num_grids, **kwargs):
        super().__init__(**kwargs)
        self.dt = dt
        self.num_grids = num_grids

        t0 = tf.constant(0.)
        self.tN = t0 + num_grids * dt

        input_dim = 28 * 28
        self._model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(input_dim, activation='relu'),
        ])
        self._model.build([None, input_dim])

        def fn(t, x):
            return self._model(x)

        self._node_fn = get_node_function(RKSolver(self.dt), 0., fn)

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
    MyLayer(dt=1e-2, num_grids=10),
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
