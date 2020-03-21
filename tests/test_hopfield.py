import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from node.core import get_node_function
from node.solvers.runge_kutta import RK4Solver
from node.hopfield import hopfield, identity


# for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


DTYPE = 'float32'
tf.keras.backend.set_floatx(DTYPE)


class MyLayer(tf.keras.layers.Layer):

  def __init__(self, units, dt, num_grids, **kwargs):
    super().__init__(**kwargs)
    self.dt = dt
    self.num_grids = num_grids

    t0 = tf.constant(0., dtype=DTYPE)
    self.tN = t0 + num_grids * dt

    self._model = tf.keras.Sequential([
        tf.keras.layers.Dense(1024, activation='relu', dtype=DTYPE),
        tf.keras.layers.Dense(1, activation='sigmoid', dtype=DTYPE),
    ])
    self._model.build([None, units])
    self._pvf = hopfield(identity, self._model)

    self._node_fn = get_node_function(RK4Solver(self.dt, dtype=DTYPE),
                                      tf.constant(0., dtype=DTYPE),
                                      self._pvf)

  def call(self, x):
    y = self._node_fn(self.tN, x)
    return y

  def get_config(self):
    return super().get_config().copy()


def process(X, y):
  X = X / 255.
  X = np.reshape(X, [-1, 28 * 28])
  y = np.eye(10)[y]
  return X.astype(DTYPE), y.astype(DTYPE)


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = process(x_train, y_train)
x_test, y_test = process(x_test, y_test)

scalar = StandardScaler()
scalar.fit(x_train)
x_train = scalar.transform(x_train)
x_test = scalar.transform(x_test)

model = tf.keras.Sequential([
    tf.keras.layers.Input([28 * 28]),
    tf.keras.layers.Dense(64, use_bias=False),  # down-sampling
    MyLayer(64, dt=1e-1, num_grids=5),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer=tf.optimizers.Adam(1e-3),
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=12, batch_size=128)
