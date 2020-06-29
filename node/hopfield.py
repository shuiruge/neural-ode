r"""
Implement the Hopfield network. Try to learn the Hebb rule (or better) using
modern SGD methods.

References
----------
1. Information Theory, Inference, and Learning Algorithms, chapter 42,
   Algorithm 42.9.
"""

import tensorflow as tf
from node.core import get_node_function, get_dynamical_node_function
from node.solvers.runge_kutta import RKF56Solver
from node.solvers.dynamical_runge_kutta import DynamicalRKF56Solver


@tf.function
def kernel_constraint(kernel):
  """Symmetric kernel with vanishing diagonal.

  Parameters
  ----------
  kernel : tensor
    Shape (N, N) for a positive integer N.

  Returns
  -------
  tensor
    The shape and dtype as the input.
  """
  w = (kernel + tf.transpose(kernel)) / 2
  w = tf.linalg.set_diag(w, tf.zeros(kernel.shape[0:-1]))
  return w


class DiscreteTimeHopfieldLayer(tf.keras.layers.Layer):
  r"""Implements the algorithm 42.9 of ref [1].

  References
  ----------
  1. Information Theory, Inference, and Learning Algorithms, chapter 42.

  Parameters
  ----------
  units : int
  activation : str or tensorflow activation, optional
  relax_tol : float, optional
    Tolerance for relaxition.
  """

  def __init__(self, units,
               activation='tanh',
               relax_tol=1e-2,
               name='DiscreteTimeHopfieldLayer',
               **kwargs):
    super().__init__(name=name, **kwargs)
    self.relax_tol = tf.convert_to_tensor(relax_tol)

    self._dense = tf.keras.layers.Dense(
      units, activation, kernel_constraint=kernel_constraint)

  def call(self, x, training=None):
    if training:
      y = self._dense(x)
    else:
      new_x = self._dense(x)
      while tf.reduce_max(tf.abs(new_x - x)) > self.relax_tol:
        x = new_x
        new_x = self._dense(x)
      y = new_x
    return y


class StopCondition:
  """Stopping condition for dynamical ODE solver.

  Attributes
  ----------
  relax_time : scalar
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
    self.max_time = tf.convert_to_tensor(max_time)
    self.relax_tol = tf.convert_to_tensor(relax_tol)

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


class ContinuousTimeHopfieldLayer(tf.keras.layers.Layer):
  r"""Implements the extension of algorithm 42.9 of ref [1], for the
  continuous-time case.

  References
  ----------
  1. Information Theory, Inference, and Learning Algorithms, chapter 42.

  Parameters
  ----------
  units : int
  activation : str or tensorflow activation, optional
  tau : float, optional
    The tau parameter in the equation (42.17) of ref [1].
  static_solver : ODESolver, optional
  dynamical_solver : DynamicalODESolver
  training_time : float, optional
    Integration time for training state.
  max_time : float, optional
    Maximum value of time that trigers the stopping condition.
  relax_tol: float, optional
    Tolerance for relaxition.
  """

  def __init__(self, units,
               activation='tanh',
               tau=1,
               static_solver=RKF56Solver(
                 dt=1e-1, tol=1e-3, min_dt=1e-2),
               dynamical_solver=DynamicalRKF56Solver(
                 dt=1e-1, tol=1e-3, min_dt=1e-2),
               training_time=1e-1,
               max_time=1e+3,
               relax_tol=1e-2,
               name='ContinuousTimeHopfieldLayer',
               **kwargs):
    super().__init__(name=name, **kwargs)
    self.training_time = tf.convert_to_tensor(training_time)

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
