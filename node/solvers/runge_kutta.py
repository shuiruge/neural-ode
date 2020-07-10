import tensorflow as tf
from node.base import Diagnostics, ODEResult, ODESolver
from node.utils.nest import nest_map


def add(phase_point, other_phase_point):
  """
  Parameters
  ----------
  phase_point : PhasePoint
  other_phase_point : PhasePoint

  Returns
  -------
  PhasePoint
  """

  @nest_map
  def _add(tensor, other_tensor):
    return tensor + other_tensor

  return _add(phase_point, other_phase_point)


def scalar_product(scalar, phase_point):
  """
  Parameters
  ----------
  scalar : scalar
  phase_point : PhasePoint

  Returns
  -------
  PhasePoint
  """

  @nest_map
  def _scalar_product(tensor):
    return scalar * tensor

  return _scalar_product(phase_point)


@tf.function
def sign(x):
  r"""Like `tf.sign`, but keeps the dtype.

  Parameters
  ----------
  x : scalar

  Returns
  -------
  scalar
  """
  if x > 0:
    return tf.constant(1, dtype=x.dtype)
  elif x == 0:
    return tf.constant(0, dtype=x.dtype)
  else:
    return tf.constant(-1, dtype=x.dtype)


class RungeKuttaStep:
  """Computes the :math:`k`s.

  ```math

  Let $N$ the order of Runge Kutta methods. Then this function computes

  $$ k_0 = dt f(t, x), $$

  and, for $i = 1 \ldots N - 1$,

  $$ k_i = dt f(t + a_i dt, x + \sum_{j = 0}^{i - 1} b_{i j} k_j). $$

  These $k$s are employed for computing $x(t + dt)$ and error estimation.

  ```

  C.f. Shampine (1986), eq. (2.1).

  Parameters
  ----------
  a : list of optional of float
    a[0] shall be `None` and others are floats.
  b : list of optional of list of float
    b[0] is `None` and others are lists of floats.
  """

  def __init__(self, a, b):
    assert len(a) == len(b)
    self.a = a
    self.b = b
    self.order = len(a)

  def __call__(self, fn, t, x, dt):
    """
    Parameters
    ----------
    fn : PhaseVectorField
    t : Time
    x : PhasePoint
    dt : Time

    Returns
    -------
    list of PhasePoint
    """

    @tf.function
    def runge_kutta_step(t, x, dt):
      k0 = scalar_product(dt, fn(t, x))
      ks = [k0]

      for i in range(1, self.order):
        ti = t + self.a[i] * dt

        @nest_map
        def xi(x, *ks):
          return x + sum(bij * kj for bij, kj in zip(self.b[i], ks))

        ki = scalar_product(dt, fn(ti, xi(x, *ks)))
        ks.append(ki)

      return ks

    return runge_kutta_step(t, x, dt)


class RungeKuttaDiagnostics(Diagnostics):

  def __init__(self):
    self.num_steps = tf.Variable(0, trainable=False)


class RungeKuttaSolver(ODESolver):
  """

  ```math
  Let $N$ the order of Runge-Kutta method, then

  $$ x(t + dt) = x(t) + \sum_{i = 0}^{N - 1} c_i k_i(t) $$
  ```

  Attributes
  ----------
  diagnostics : RungeKuttaDiagnostics

  Parameters
  ----------
  a : list of optional of float
    a[0] shall be `None` and others are floats.
  b : list of optional of list of float
    b[0] is `None` and others are lists of floats.
  c : list of float
  dt : float
  min_dt : float
  dtype : string or dtype, optional
  """

  def __init__(self, a, b, c, dt, min_dt, dtype='float32'):
    assert min_dt < dt
    self.c = c
    self.dt = tf.convert_to_tensor(dt, dtype=dtype)
    self.min_dt = tf.convert_to_tensor(min_dt, dtype=dtype)
    self._rk_step = RungeKuttaStep(a, b)

    self.diagnostics = RungeKuttaDiagnostics()

  def __call__(self, fn):

    @nest_map
    def dx(*ks):
      return sum(ci * ki for ci, ki in zip(self.c, ks))

    @tf.function
    def forward(t0, t1, x0):
      s = sign(t1 - t0)
      t = t0
      x = x0
      dt = s * self.dt

      while s * (t1 - t) > self.min_dt:
        if tf.abs(t1 - t) < tf.abs(dt):
          dt = t1 - t
        ks = self._rk_step(fn, t, x, dt)
        x = add(x, dx(*ks))
        t = t + dt
        self.diagnostics.num_steps.assign_add(1)

      return ODEResult(t1, x)

    return forward


def l2_norm(x):
  """L_2 norm."""
  return tf.sqrt(tf.reduce_sum(tf.square(x)))


def rms_norm(x):
  """RMS norm."""
  return tf.sqrt(tf.reduce_mean(tf.square(x)))


def l_inf_norm(x):
  """L_{\infinity} norm."""
  return tf.reduce_max(tf.abs(x))


def norm(x):
  # XXX: Reduce_max? Without considering the batch-dimension?
  # Because the dt is universal for all samples.
  flat = tf.nest.flatten(x)
  # use L-infinity norm since it's dimension independent
  return tf.reduce_max([l_inf_norm(_) for _ in flat])


class RungeKuttaFehlbergDiagnostics:

  def __init__(self):
    self.num_steps = tf.Variable(0, trainable=False)
    self.num_accepted = tf.Variable(0, trainable=False)
    self.succeed = tf.Variable(True, trainable=False)
    self.total_error = tf.Variable(0., trainable=False)


class RungeKuttaFehlbergSolver(ODESolver):
  r"""
  ```math

  Let $R$ denote the error and $N$ the order of Runge-Kutta method, then

  $$ R = \| \sum_{i = 0}^{N - 1} e_i k_i \| / dt, $$

  where $e$s are constants.

  ```

  References
  ----------
  Numerical Analysis by Burden and Faires, section 5.5, algorithm 5.3, p297.

  Attributes
  ----------
  diagnostics : RungeKuttaFehlbergDiagnostics

  Parameters
  ----------
  a : list of optional of float
    a[0] shall be `None` and others are floats.
  b : list of optional of list of float
    b[0] is `None` and others are lists of floats.
  c : list of float
  e : list of float
  init_dt : float
  tol : float
  min_dt : float
  max_dt: optional of float
  dtype : string or dtype, optional
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

    self.diagnostics = RungeKuttaFehlbergDiagnostics()

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
      s = sign(t1 - t0)
      t = t0
      x = x0
      dt = s * self.init_dt

      while s * (t1 - t) > self.min_dt:
        if tf.abs(t1 - t) < tf.abs(dt):
          dt = t1 - t

        ks = self._rk_step(fn, t, x, dt)
        r = error(dt, *ks)
        self.diagnostics.total_error.assign_add(r)

        # if r < self.tol:  # TODO
        if r < self.tol or tf.abs(dt) <= self.min_dt:
          x = add(x, dx(*ks))
          t = t + dt
          self.diagnostics.num_accepted.assign_add(1)

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
          self.diagnostics.succeed.assign(False)

        self.diagnostics.num_steps.assign_add(1)

      return ODEResult(t1, x)

    return forward


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
