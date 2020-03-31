"""
Study if the non-convergence is caused by the sigmoid-like (i.e. softmax)
activation in the output layer. By experiments, we find that, even though
the activation in the output layer is altered to be non-sigmoid-like, the
non-convergence stands still.
"""

import numpy as np
import tensorflow as tf
from argparse import ArgumentParser
from node.core import get_node_function
from node.solvers.runge_kutta import RK4Solver, RKF56Solver
from node.utils.initializers import GlorotUniform


# for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class MyLayer(tf.keras.layers.Layer):

  def __init__(self, units, dt, num_grids, **kwargs):
    super().__init__(**kwargs)
    self.dt = dt
    self.num_grids = num_grids

    t0 = tf.constant(0.)
    self.tN = t0 + num_grids * dt

    self._model = tf.keras.Sequential([
        tf.keras.layers.Dense(1024, activation='relu',
                              kernel_initializer=GlorotUniform(1e-1)),
        tf.keras.layers.Dense(units, activation='relu',
                              kernel_initializer=GlorotUniform(1e-1)),
    ])
    self._model.build([None, units])
    self._pvf = lambda _, x: self._model(x)

    solver = RK4Solver(self.dt)

    self._node_fn = get_node_function(
        solver, tf.constant(0.), self._pvf)

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


def create_model(num_grids):
  model = tf.keras.Sequential([
      tf.keras.layers.Input([28 * 28]),
      tf.keras.layers.Dense(64, activation='relu'),
      MyLayer(64, dt=1e-1, num_grids=num_grids),
      tf.keras.layers.Dense(10),
  ])

  model.compile(
      optimizer=tf.keras.optimizers.Adam(),
      loss='mse',
      metrics=['accuracy'])
  return model


model = create_model(num_grids=10)
model.summary()
model.fit(x_train, y_train,
          epochs=5,
          validation_data=(x_test, y_test))  # => val_acc: 0.963

eval_model = create_model(num_grids=100)
eval_model.set_weights(model.get_weights())
eval_model.evaluate(x_test, y_test)  # => acc: 0.324
