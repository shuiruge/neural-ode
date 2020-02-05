"""Implementation with the aid of the `tensorflow_probability.math.ode`
module."""

import tensorflow_probability as tfp
from node.base import ODESolver


class RKF45Solver(ODESolver):

  def __init__(self, **kwargs):
    self._solver = tfp.math.ode.DormandPrince(**kwargs)

  def __call__(self, fn):

    def forward(t0, t1, x0):
      result = self._solver.solve(fn, t0, x0, [t1])
      return result.states

    return forward


def get_node_function(solver, t0, fn):
  forward = solver(fn)

  def node_fn(t, x):
    return forward(t0, t, x)

  return node_fn


def node_wrapper(solver, t0):
  """Returns a decorator."""
  return lambda fn: get_node_function(solver, t0, fn)


def get_node_method(solver, t0, method):

  def node_method(obj, t, x):

    def fn(t, x):
      return method(obj, t, x)

    node_fn = get_node_function(solver, t0, fn)
    return node_fn(t, x)

  return node_method


def node_method_wrapper(solver, t0):
  """Returns a decorator."""
  return lambda method: get_node_method(solver, t0, method)
