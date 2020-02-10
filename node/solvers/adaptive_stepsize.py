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
      return result.states[0]

    return forward
