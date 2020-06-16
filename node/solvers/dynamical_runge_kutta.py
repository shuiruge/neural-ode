import tensorflow as tf
from node.base import DynamicalODESolver
from node.utils.nest import nest_map
from node.solvers.runge_kutta import (
  add, norm, sign, RungeKuttaStep, RungeKuttaFehlbergDiagnostics,
  RungeKuttaSolver, RungeKuttaFehlbergSolver)


class DynamicalRungeKuttaSolver(DynamicalODESolver):
  r"""

  ```math
  Let $N$ the order of Runge-Kutta method, then

  $$ x(t + dt) = x(t) + \sum_{i = 0}^{N - 1} c_i k_i(t) $$
  ```

  Args:
    a: List[Optional[float]]
      a[0] shall be `None` and others are floats.
    b: List[List[Optional[float]]]
      b[0] is `None` and others are lists of floats.
    c: List[float]
    dt: float
    min_dt: float
  """

  def __init__(self, a, b, c, dt, min_dt, dtype='float32'):
    assert min_dt < dt
    self.c = c
    self.dt = tf.convert_to_tensor(dt, dtype=dtype)
    self.min_dt = tf.convert_to_tensor(min_dt, dtype=dtype)
    self._rk_step = RungeKuttaStep(a, b)

  def __call__(self, fn, stop_condition):

    @nest_map
    def dx(*ks):
      return sum(ci * ki for ci, ki in zip(self.c, ks))

    @tf.function
    def forward(t0, x0, reverse=False):
      t = t0
      x = x0
      dt = -self.dt if reverse else self.dt
      while not stop_condition(t0, x0, t, x):
        ks = self._rk_step(fn, t, x, dt)
        x = add(x, dx(*ks))
        t = t + dt
      return (t, x)

    return forward


class DynamicalRungeKuttaFehlbergSolver(DynamicalODESolver):
  r"""
  ```math

  Let $R$ denote the error and $N$ the order of Runge-Kutta method, then

  $$ R = \| \sum_{i = 0}^{N - 1} e_i k_i \| / dt, $$

  where $e$s are constants.

  ```

  References:
    Numerical Analysis by Burden and Faires, section 5.5, algorithm 5.3, p297.

  Args:
    a: List[Optional[float]]
      a[0] shall be `None` and others are floats.
    b: List[List[Optional[float]]]
      b[0] is `None` and others are lists of floats.
    c: List[float]
    e: List[float]
    init_dt: float
    tol: float
    min_dt: float
    max_dt: Optional[float]
  """

  def __init__(self, a, b, c, e, init_dt, tol, min_dt, max_dt, dtype='float32'):
    assert len(c) == len(e)
    self.c = c
    self.e = e
    self.init_dt = tf.convert_to_tensor(init_dt, dtype=dtype)
    self.tol = tf.convert_to_tensor(tol, dtype=dtype)
    self.min_dt = tf.convert_to_tensor(min_dt, dtype=dtype)
    if max_dt is None:
      self.max_dt = None
    else:
      self.max_dt = tf.convert_to_tensor(max_dt, dtype=dtype)
    self._rk_step = RungeKuttaStep(a, b)
    self._diagnostics = RungeKuttaFehlbergDiagnostics()

  def __call__(self, fn, stop_condition):

    @nest_map
    def dx(*ks):
      return sum(ci * ki for ci, ki in zip(self.c, ks))

    def error(dt, *ks):

      @nest_map
      def _rs(*ks):
        return sum(ei * ki for ei, ki in zip(self.e, ks)) / dt

      return norm(_rs(*ks))

    @tf.function
    def forward(t0, x0, reverse=False):
      s = tf.constant(-1.) if reverse else tf.constant(1.)
      t = t0
      x = x0
      dt = -self.init_dt if reverse else self.init_dt
      s = sign(dt)

      succeed = True
      self._diagnostics.reset()

      while not stop_condition(t0, x0, t, x):
        accepted = False

        ks = self._rk_step(fn, t, x, dt)
        r = error(dt, *ks)

        # if r < self.tol:  # TODO
        if r < self.tol or tf.abs(dt) <= self.min_dt:
          accepted = True
          x = add(x, dx(*ks))
          t = t + dt

        delta = 0.84 * tf.pow(self.tol / r, 1 / 4)
        if delta < 0.1:
          dt = 0.1 * dt
        elif delta > 4:
          dt = 4 * dt
        else:
          dt = delta * dt
        if self.max_dt is not None and tf.abs(dt) > self.max_dt:
          dt = s * self.max_dt
        # Assertion is temporally not well supported in TF,
        # so we currently limit the `dt` by `self.min_dt`
        # instead of raising an error.
        if tf.abs(dt) < self.min_dt:
          dt = s * self.min_dt
          succeed = False

        self._diagnostics.update(accepted, succeed, r)
      return (t, x)

    return forward

  @property
  def diagnostics(self):
    return self._diagnostics


class DynamicalRK4Solver(DynamicalRungeKuttaSolver):

  _A = [None, 1 / 2, 1 / 2, 1]
  _B = [
      None,
      [1 / 2],
      [0, 1 / 2],
      [0, 0, 1]
  ]
  _C = [1 / 6, 1 / 3, 1 / 3, 1 / 6]

  def __init__(self, dt, min_dt=1e-3, **kwargs):
    super().__init__(self._A, self._B, self._C, dt, min_dt, **kwargs)


class DynamicalRKF56Solver(DynamicalRungeKuttaFehlbergSolver):

  _A = [None, 1 / 4, 3 / 8, 12 / 13, 1, 1 / 2]
  _B = [
      None,
      [1 / 4],
      [3 / 32, 9 / 32],
      [1932 / 2197, -7200 / 2197, 7296 / 2197],
      [439 / 216, -8, 3680 / 513, -845 / 4104],
      [-8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40]
  ]
  _C = [25 / 216, 0, 1408 / 2565, 2197 / 4104, -1 / 5, 0]
  _E = [1 / 360, 0, -128 / 4275, -2197 / 75240, 1 / 50, 2 / 55]

  def __init__(self, dt, tol, min_dt, max_dt=None, **kwargs):
    super().__init__(self._A, self._B, self._C, self._E,
                     dt, tol, min_dt, max_dt, **kwargs)
