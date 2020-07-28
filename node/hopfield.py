"""
Implement the Hopfield network. Try to learn the Hebb rule (or better) using
modern SGD methods. C.f. algorithm 42.9 of ref [1]_.

References
----------
.. [1] D. Mackay, "Information Theory, Inference, and Learning Algorithms".
"""

import tensorflow as tf

from node.core import get_dynamical_node_function
from node.solvers.dynamical_runge_kutta import DynamicalRKF56Solver
from node.solvers.runge_kutta import RKF56Solver
from node.utils.binary import binarize, inverse_binarize


def _binarize(x, num_bits):
  """
  Parameters
  ----------
  x : tensor
  num_bits : optional of int

  Returns
  -------
  tensor
    Shape: `x.shape[:-1] + [num_bits * x.shape[-1]]`, and values in {-1, +1}.
    if `num_bits` is `None`, then returns `x` directly.
  """
  if num_bits is None:
    return x
  y = binarize(x, num_bits)
  shape = y.get_shape().as_list()
  shape = [-1 if n is None else n for n in shape]
  flatten_shape = shape[:-2] + [shape[-2] * shape[-1]]
  return tf.reshape(y, flatten_shape)


def _inverse_binarize(x, num_bits):
  """
  Parameters
  ----------
  x : tensor
  num_bits : optional of int

  Returns
  -------
  tensor
    Shape: `x.shape[:-1] + [x.shape[-1] // num_bits]`, and values in [-1, +1].
    if `num_bits` is `None`, then returns `x` directly.
  """
  if num_bits is None:
    return x
  shape = x.get_shape().as_list()
  shape = [-1 if n is None else n for n in shape]
  nested_shape = shape[:-1] + [shape[-1] // num_bits, num_bits]
  x = tf.reshape(x, nested_shape)
  return inverse_binarize(x)


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
  r"""Implements the algorithm 42.9 of ref [1]_.

  References
  ----------
  .. [1] D. Mackay, "Information Theory, Inference, and Learning Algorithms".

  Parameters
  ----------
  activation : str or tensorflow_activation, optional
    Maps onto [-1, 1].
  relax_tol : float, optional
    Relative tolerance for relaxition.
  reg_factor: float, optional
  """

  def __init__(self,
               activation='tanh',
               relax_tol=1e-2,
               reg_factor=0,
               num_bits=None,
               **kwargs):
    super().__init__(**kwargs)
    self.activation = activation
    self.relax_tol = tf.convert_to_tensor(relax_tol)
    self.reg_factor = reg_factor
    self.num_bits = num_bits

  def build(self, input_shape):
    units = input_shape[-1]
    if self.num_bits is not None:
      units *= self.num_bits

    self._f = tf.keras.layers.Dense(
      units, self.activation, kernel_constraint=kernel_constraint)

    super().build(input_shape)

  def call(self, x, training=None):
    x = _binarize(x, self.num_bits)

    if training:
      y = self._f(x)
    else:
      new_x = self._f(x)
      while tf.reduce_max(tf.abs(new_x - x)) > self.relax_tol:
        x = new_x
        new_x = self._f(x)
      y = new_x

    loss = tf.reduce_mean(tf.abs(y - x))
    self.add_loss(self.reg_factor * loss, inputs=True)

    y = _inverse_binarize(y, self.num_bits)
    return y


class StopCondition:
  """Stopping condition for dynamical ODE solver.

  Attributes
  ----------
  relax_time : scalar
    The time when relax. Being `-1` means that `self.max_time` is reached
    before relaxing.

  Parameters
  ----------
  pvf : phase_vector_field
  max_time : float
    Returns `True` when `t1 - t0 > max_time`.
  relax_tol : float
    Relative tolerance for relaxition.
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
  r"""Implements the extension of algorithm 42.9 of ref [1]_, for the
  continuous-time case.

  Notes
  -----
  Argument `zero_diag` is default to `True`. When setting it `False`, weight-
  regularization shall be added, so as to avoid learning an identity transform,
  which has been observed in experiments when weight-regularization is absent.

  References
  ----------
  .. [1] D. Mackay, "Information Theory, Inference, and Learning Algorithms".

  Parameters
  ----------
  activation : str or tensorflow_activation, optional
    Maps onto [-1, 1].
  tau : float, optional
    The tau parameter in the equation (42.17) of ref [1]_.
  static_solver : ODESolver, optional
  dynamical_solver : DynamicalODESolver
  max_time : float, optional
    Maximum value of time that trigers the stopping condition.
  relax_tol : float, optional
    Relative tolerance for relaxition.
  reg_factor: float, optional
    The factor of the regularization-loss.
  """

  def __init__(self,
               activation='tanh',
               tau=1,
               static_solver=RKF56Solver(
                 dt=1e-1, tol=1e-3, min_dt=1e-2),
               dynamical_solver=DynamicalRKF56Solver(
                 dt=1e-1, tol=1e-3, min_dt=1e-2),
               max_time=1e+3,
               relax_tol=1e-3,
               reg_factor=0,
               num_bits=None,
               **kwargs):
    super().__init__(**kwargs)
    self.activation = activation
    self.tau = tau
    self.static_solver = static_solver
    self.dynamical_solver = dynamical_solver
    self.max_time = max_time
    self.relax_tol = relax_tol
    self.reg_factor = reg_factor
    self.num_bits = num_bits

  def build(self, input_shape):
    units = input_shape[-1]
    if self.num_bits is not None:
      units *= self.num_bits

    f = tf.keras.layers.Dense(
      units, self.activation, kernel_constraint=kernel_constraint)

    def pvf(t, x):
      return (-x + f(x)) / self.tau

    stop_condition = StopCondition(pvf, self.max_time, self.relax_tol)

    self._f = f
    self._pvf = pvf
    self._stop_condition = stop_condition
    self._node_f = get_dynamical_node_function(
      self.dynamical_solver, self.static_solver, pvf, stop_condition)

    super().build(input_shape)

  def call(self, x, training=None):
    x = _binarize(x, self.num_bits)

    t0 = tf.constant(0.)
    if training:
      y = self._f(x)
    else:
      y = self._node_f(t0, x)

    loss = tf.reduce_mean(tf.abs(y - x))
    self.add_loss(self.reg_factor * loss)

    y = _inverse_binarize(y, self.num_bits)
    return y
