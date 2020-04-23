import tensorflow as tf
from node.base import DynamicalODESolver
from node.solvers.runge_kutta import RungeKuttaStep, add


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
      while not stop_condition(t, x):
        ks = self._rk_step(fn, t, x, dt)
        x = add(x, dx(*ks))
        t = t + dt
      return (t, x)

    return forward


def norm(x):

  def l2(x):
    return tf.sqrt(tf.reduce_sum(tf.square(x)))

  flat = tf.nest.flatten(x)
  return tf.reduce_max([l2(_) for _ in flat])


class RungeKuttaFehlbergSolver(ODESolver):
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

  def __call__(self, fn):

    @nest_map
    def dx(*ks):
      return sum(ci * ki for ci, ki in zip(self.c, ks))

    def error(dt, *ks):

      @nest_map
      def _rs(*ks):
        return sum(ei * ki for ei, ki in zip(self.e, ks)) / dt

      return norm(_rs(*ks))

    @tf.function
    def forward(t0, t1, x0):
      # If t0 > t1, then flip the t-axis to ensure the
      # situation of the Runge-Kutta method
      flip = False
      if t0 > t1:
        t0, t1 = -t0, -t1
        flip = True

      t = t0
      x = x0
      dt = self.init_dt
      succeed = True
      self._diagnostics.reset()

      while t1 - t > self.min_dt:
        accepted = False

        if t < t1 and t + dt > t1:
          dt = t1 - t

        ks = self._rk_step(fn, t, x, dt, flip=flip)
        r = error(dt, *ks)

        # if r < self.tol:  # TODO
        if r < self.tol or dt <= self.min_dt:
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
        if self.max_dt is not None and dt > self.max_dt:
          dt = self.max_dt
        # Assertion is temporally not well supported in TF,
        # so we currently limit the `dt` by `self.min_dt`
        # instead of raising an error.
        if dt < self.min_dt:
          dt = self.min_dt
          succeed = False

        self._diagnostics.update(accepted, succeed, r)
      return x

    return forward

  @property
  def diagnostics(self):
    return self._diagnostics


class RungeKuttaFehlbergDiagnostics:
  
  def __init__(self):
    self.num_steps = tf.Variable(0, trainable=False)
    self.num_accepted = tf.Variable(0, trainable=False)
    self.accept_ratio = tf.Variable(0., trainable=False)
    self.succeed = tf.Variable(True, trainable=False)
    self.total_error = tf.Variable(0., trainable=False)

  def update(self, accepted, succeed, error):
    self.num_steps.assign_add(1)
    if accepted:
      self.num_accepted.assign_add(1)
    self.accept_ratio.assign(
        tf.cast(self.num_accepted, self.accept_ratio.dtype) /
        tf.cast(self.num_steps, self.accept_ratio.dtype))
    if not succeed:
      self.succeed.assign(False)
    self.total_error.assign_add(error)
  
  def reset(self):
    self.num_steps.assign(0)
    self.num_accepted.assign(0)
    self.accept_ratio.assign(0.)
    self.succeed.assign(True)
    self.total_error.assign(0.)

  def __repr__(self):
    return ('num_steps: {}, num_accepted: {}, accept_ratio: {}, '
            'succeed: {}, total_error: {}').format(
                self.num_steps, self.num_accepted, self.accept_ratio,
                self.succeed, self.total_error)


class RK4Solver(RungeKuttaSolver):

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


class RKF56Solver(RungeKuttaFehlbergSolver):

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


class ODESolveError(Exception):
  pass
