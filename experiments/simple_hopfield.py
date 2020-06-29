r"""
Description
-----------
Implement the Hopfield network. Try to learn the Hebb rule (or better) using
modern SGD methods.

References
----------
1. Information Theory, Inference, and Learning Algorithms, chapter 42,
   Algorithm 42.9.
"""

import numpy as np
import tensorflow as tf
from node.core import get_dynamical_node_function, get_node_function
from node.solvers.runge_kutta import RK4Solver, RKF56Solver
from node.solvers.dynamical_runge_kutta import (
  DynamicalRK4Solver, DynamicalRKF56Solver)


# for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# limit CPU usage
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

tf.keras.backend.clear_session()

IMAGE_SIZE = (16, 16)
NUM_MEMORY = 100
FLIP_RATIO = 0.2

# for continous-time Hopfield
# SOLVER = 'rk4'
SOLVER = 'rkf56'
TAU = 1
T = 0.1
MAX_T = 1000
RELAX_TOL = 1e-2
DEBUG = 0


def kernel_constraint(kernel):
  w = (kernel + tf.transpose(kernel)) / 2
  w = tf.linalg.set_diag(w, tf.zeros(kernel.shape[0:-1]))
  return w


class HopfieldLayer(tf.keras.layers.Layer):
  r"""
  Description
  -----------
  Implements the algorithm 42.9 of ref [1].

  References
  ----------
  1. Information Theory, Inference, and Learning Algorithms, chapter 42.
  """

  def __init__(self, units, activation, name='HopfieldLayer', **kwargs):
    super().__init__(name=name, **kwargs)
    self._dense = tf.keras.layers.Dense(
      units, activation, kernel_constraint=kernel_constraint)

  def call(self, x):
    return self._dense(x)


@tf.custom_gradient
def boundary_reflect(x):  # TODO: fix the bug at boundaries.
  r"""
  Description
  -----------
  Inverse of $G$ function in the reference [1].

  $$ G^{-1}: \mathbb{R}^n \mapsto [0, 1]^n $$

  References
  ----------
  1. Diffusions for Global Optimization, S. Geman and C. Hwang
  """
  int_i = tf.cast(x, 'int32')
  i = tf.cast(int_i, x.dtype)
  delta = x - i
  y = tf.where(int_i % 2 == 0, x, i + 1 - x)

  def grad_fn(dy):
    return tf.where(int_i % 2 == 0, dy, -dy)

  return y, grad_fn


class StopCondition:

  def __init__(self, pvf):
    self.pvf = pvf
    self.relax_time = tf.Variable(-1., trainable=False)

  def __call__(self, t0, x0, t1, x1):
    if t1 - t0 > MAX_T:
      self.relax_time.assign(-1)
      return True
    if tf.reduce_max(tf.abs(self.pvf(t1, x1))) < RELAX_TOL:
      self.relax_time.assign(t1)
      return True
    return False


class ContinousHopfieldLayer(tf.keras.layers.Layer):
  r"""
  Description
  -----------
  Implements the extension of algorithm 42.9 of ref [1], for the continous-time
  case.

  References
  ----------
  1. Information Theory, Inference, and Learning Algorithms, chapter 42.
  """

  def __init__(self, units, activation, name='HopfieldLayer', **kwargs):
    super().__init__(name=name, **kwargs)
    self._dense = tf.keras.layers.Dense(
      units, activation, kernel_constraint=kernel_constraint)

    if SOLVER == 'rk4':
      solver = RK4Solver(0.1)
      dynamical_solver = DynamicalRK4Solver(0.1)
    elif SOLVER == 'rkf56':
      solver = RKF56Solver(0.1, tol=1e-3, min_dt=1e-2)
      dynamical_solver = DynamicalRKF56Solver(0.1, tol=1e-3, min_dt=1e-2)
    else:
      raise ValueError()

    def pvf(_, x):
      """C.f. section 42.6 of ref [1]."""
      return (-x + self._dense(x)) / TAU

    self.pvf = pvf
    self.stop_condition = StopCondition(self.pvf)
    self.static_node_fn = get_node_function(solver, self.pvf)
    self.dynamical_node_fn = get_dynamical_node_function(
      dynamical_solver, solver, self.pvf, self.stop_condition)

  def call(self, x, training=None):
    t0 = tf.constant(0.)
    if training:
      t1 = tf.constant(T)
      return self.static_node_fn(t0, t1, x)
    else:
      return self.dynamical_node_fn(t0, x)


def pooling(X, size):
  X = np.expand_dims(X, axis=-1)
  X = tf.image.resize(X, size).numpy()
  return X


def process_data(X, y):
  X = X / 255
  X = pooling(X, IMAGE_SIZE)
  X = np.reshape(X, [-1, IMAGE_SIZE[0] * IMAGE_SIZE[1]])
  X = np.where(X < 0.5, -1, 1)
  y = np.eye(10)[y]
  return X.astype('float32'), y.astype('float32')


mnist = tf.keras.datasets.mnist
(x_train, y_train), _ = mnist.load_data()
x_train, y_train = process_data(x_train, y_train)
x_train = x_train[:NUM_MEMORY]


def train(hopfield, x_train, epochs=500):
  model = tf.keras.Sequential([
    hopfield,
    tf.keras.layers.Lambda(lambda x: x / 2 + 0.5),
  ])
  model.compile(loss='binary_crossentropy', optimizer='adam')
  y_train = x_train / 2 + 0.5
  model.fit(x_train, y_train, epochs=epochs, verbose=2)


def show_denoising_effect(hopfield):
  X = tf.constant(x_train)
  noised_X = tf.where(tf.random.uniform(shape=X.shape) < FLIP_RATIO,
                      1 - X, X)
  X_star = noised_X
  if isinstance(hopfield, HopfieldLayer):
    for i in range(100):
      X_star = hopfield(X_star)
  elif isinstance(hopfield, ContinousHopfieldLayer):
    X_star = hopfield(X_star)
    print('Relaxed at:', hopfield.stop_condition.relax_time)
  else:
    raise ValueError()

  print(tf.nn.moments(tf.abs(noised_X - X), axes=[0, 1]))
  print(tf.nn.moments(tf.abs(X_star - X), axes=[0, 1]))
  print(tf.reduce_max(tf.abs(noised_X - X)))
  print(tf.reduce_max(tf.abs(X_star - X)))


units = IMAGE_SIZE[0] * IMAGE_SIZE[1]
# hopfield = HopfieldLayer(units, 'tanh')
hopfield = ContinousHopfieldLayer(units, 'tanh')
train(hopfield, x_train)
show_denoising_effect(hopfield)
