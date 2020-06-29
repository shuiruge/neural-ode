r"""
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
MEMORY_SIZE = 100
FLIP_RATIO = 0.2

# for discrete-time Hopfield
NUM_RECURTION = 1000

# for continous-time Hopfield
# SOLVER = 'rk4'
SOLVER = 'rkf56'
DT = 1e-1
SOLVER_TOL = 1e-3
MIN_DT = 1e-2
TAU = 1e+0
TRAINING_T = 1e-1
MAX_T = 1e+3
RELAX_TOL = 1e-2


@tf.function
def kernel_constraint(kernel):
  w = (kernel + tf.transpose(kernel)) / 2
  w = tf.linalg.set_diag(w, tf.zeros(kernel.shape[0:-1]))
  return w


class HopfieldLayer(tf.keras.layers.Layer):
  r"""Implements the algorithm 42.9 of ref [1].

  References
  ----------
  1. Information Theory, Inference, and Learning Algorithms, chapter 42.

  Parameters
  ----------
  units : int
  activation : str or tensorflow activation
  """

  def __init__(self, units, activation, name='HopfieldLayer', **kwargs):
    super().__init__(name=name, **kwargs)
    self._dense = tf.keras.layers.Dense(
      units, activation, kernel_constraint=kernel_constraint)

  def call(self, x):
    return self._dense(x)


@tf.custom_gradient
def boundary_reflect(x):  # TODO: fix the bug at boundaries.
  """Inverse of $G$ function in the reference [1].

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
  """Stopping condition for dynamical ODE solver.

  Attributes
  ----------
  relax_time : tf.Variable
    The time when relax.

  Parameters
  ----------
  pvf : phase_vector_field
  max_time : float
  relax_tol : float
    Tolerance for relaxition.
  """

  def __init__(self, pvf, max_time, relax_tol):
    self.pvf = pvf
    self.max_time = tf.constant(max_time)
    self.relax_tol = tf.constant(relax_tol)

    self.relax_time = tf.Variable(0., trainable=False)

  @tf.function
  def __call__(self, t0, x0, t1, x1):
    if t1 - t0 > self.max_time:
      self.relax_time.assign(-1)
      return True
    if tf.reduce_max(tf.abs(self.pvf(t1, x1))) < self.relax_tol:
      self.relax_time.assign(t1)
      return True
    return False


class ContinousHopfieldLayer(tf.keras.layers.Layer):
  r"""Implements the extension of algorithm 42.9 of ref [1], for the
  continous-time case.

  References
  ----------
  1. Information Theory, Inference, and Learning Algorithms, chapter 42.

  Parameters
  ----------
  units : int
  activation : str or tensorflow activation
  tau : float
    The tau parameter in the equation (42.17) of ref [1].
  static_solver : ODESolver
  dynamical_solver : DynamicalODESolver
  training_time : float
    Integration time for training state.
  max_time : float
    Maximum value of time that trigers the stopping condition.
  relax_tol: float
    Tolerance for relaxition.
  """

  def __init__(self, units, activation, tau,
               static_solver, dynamical_solver,
               training_time, max_time, relax_tol,
               name='ContinousHopfieldLayer', **kwargs):
    super().__init__(name=name, **kwargs)
    self.training_time = tf.constant(training_time)

    self._dense = tf.keras.layers.Dense(
      units, activation, kernel_constraint=kernel_constraint)

    def pvf(_, x):
      """C.f. section 42.6 of ref [1]."""
      return (-x + self._dense(x)) / tau

    self.pvf = pvf
    self.stop_condition = StopCondition(self.pvf, max_time, relax_tol)
    self.static_node_fn = get_node_function(static_solver, self.pvf)
    self.dynamical_node_fn = get_dynamical_node_function(
      dynamical_solver, static_solver, self.pvf, self.stop_condition)

  def call(self, x, training=None):
    t0 = tf.constant(0.)
    if training:
      y = self.static_node_fn(t0, self.training_time, x)
    else:
      y = self.dynamical_node_fn(t0, x)
    return y


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


def train(hopfield, x_train, epochs=500):
  model = tf.keras.Sequential([
    hopfield,
    tf.keras.layers.Lambda(lambda x: x / 2 + 0.5),
  ])
  model.compile(loss='binary_crossentropy', optimizer='adam')
  y_train = x_train / 2 + 0.5
  model.fit(x_train, y_train, epochs=epochs, verbose=2)


def show_denoising_effect(hopfield, X):
  X = tf.convert_to_tensor(X)
  noised_X = tf.where(tf.random.uniform(shape=X.shape) < FLIP_RATIO,
                      1 - X, X)
  X_star = noised_X
  if isinstance(hopfield, HopfieldLayer):
    for i in range(NUM_RECURTION):
      X_star = hopfield(X_star)
  elif isinstance(hopfield, ContinousHopfieldLayer):
    X_star = hopfield(X_star)
    tf.print('relaxed at:', hopfield.stop_condition.relax_time)
  else:
    raise ValueError()

  print(tf.nn.moments(tf.abs(noised_X - X), axes=[0, 1]))
  print(tf.nn.moments(tf.abs(X_star - X), axes=[0, 1]))
  print(tf.reduce_max(tf.abs(noised_X - X)))
  print(tf.reduce_max(tf.abs(X_star - X)))


def create_hopfield_layer(is_continous_time):
  units = IMAGE_SIZE[0] * IMAGE_SIZE[1]
  if not is_continous_time:
    hopfield = HopfieldLayer(units, 'tanh')
  else:
    if SOLVER == 'rk4':
      solver = RK4Solver(DT)
      dynamical_solver = DynamicalRK4Solver(DT)
    elif SOLVER == 'rkf56':
      solver = RKF56Solver(DT, tol=SOLVER_TOL, min_dt=MIN_DT)
      dynamical_solver = DynamicalRKF56Solver(DT, tol=SOLVER_TOL, min_dt=MIN_DT)
    else:
      raise ValueError()
    hopfield = ContinousHopfieldLayer(
      units, 'tanh', TAU, solver, dynamical_solver, TRAINING_T, MAX_T,
      RELAX_TOL)
  return hopfield


mnist = tf.keras.datasets.mnist
(x_train, y_train), _ = mnist.load_data()
x_train, y_train = process_data(x_train, y_train)
x_train = x_train[:MEMORY_SIZE]

hopfield = create_hopfield_layer(is_continous_time=True)
train(hopfield, x_train)
show_denoising_effect(hopfield, x_train)
