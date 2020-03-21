# TODO

import tensorflow as tf
import tensorflow_probability as tfp
from node.core import get_node_function
from node.solvers.runge_kutta import RK4Solver


class CNF(tfp.bijectors.Bijector):

  def __init__(self, t0, t1, solver=RK4Solver(1e-1),
               validate_args=False, name='cnf'):
    super().__init__(forward_min_event_ndims=1,
                     validate_args=validate_args,
                     name=name)

    self.t0 = tf.convert_to_tensor(t0)
    self.t1 = tf.convert_to_tensor(t1)
    self.solver = solver

    self._forward_fn = get_node_function(
        self.solver, self.t0, self._dynamics)
    self._inverse_fn = get_node_function(
        self.solver, self.t1, self._dynamics)
    self._forward_log_prob_fn = get_node_function(
        self.solver, self.t0, self._log_prob_dynamics)

  def _dynamics(self, t, x):
    return NotImplemented

  def _log_prob_dynamics(self, t, x_and_log_prob):
    return NotImplemented

  def _forward(self, x):
    return self._forward_fn(self.t1, x)

  def _inverse(self, y):
    return self._inverse_fn(self.t0, y)

  def _forward_log_det_jacobian(self, x):
    x_and_log_det_jacobian = [x, tf.zeros(x.shape[:-1])]
    x, log_det_jacobian = self._forward_log_prob_fn(
        self.t1, x_and_log_det_jacobian)
    return log_det_jacobian
