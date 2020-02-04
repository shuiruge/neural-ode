import tensorflow as tf
from node.base import ODESolver
from node.utils.nest import nest_map


class FixGridODESolver(ODESolver):
    r"""ODE solver with fix grid methods.

    C.f. the slides of Neural ODE in the deep-learning-notes project
    (GitHub @kmkolasinski).

    Args:
        step_fn:
            Step-function for fixed grid integration method, e.g. Euler's
            method. The input is the phase vector field. Returns a callable
            that makes one step forward, arguments for time, time difference,
            and current phase point, and output for the next phase point.
        dt: float

    Returns: ODESolver
    """

    def __init__(self, step_fn, dt, dtype=tf.float32):
        self.step_fn = step_fn
        self.dt = tf.convert_to_tensor(dt, dtype=dtype)
        self.dtype = dtype

    def __call__(self, fn):
        step_forward = self.step_fn(fn)
        cast = _cast(self.dtype)

        @tf.function
        def forward(t0, t1, x0):
            cast_back = _cast(t0.dtype)

            t0 = cast(t0)
            t1 = cast(t1)
            x = cast(x0)

            if t1 == t0:
                return cast_back(x)

            if tf.abs(t1 - t0) < self.dt:
                x = step_forward(t0, t1 - t0, x)
                return cast_back(x)

            # TODO: illustrate why the `dt` shall be computed as so
            dt = tf.where(t1 > t0, self.dt, -self.dt)

            # To compute the `ts`, it's better to use `tf.linspace` instead
            # of using `tf.range`, since `tf.range(t0, tN, dt)` will raise
            # error when `t0 > tN`, which will happen in the backward process.
            # However, `tf.linspace` works well in this case.
            N = int((t1 - t0) / dt + 1)

            for t in tf.linspace(t0, t1, N):
                x = step_forward(t, dt, x)

            return cast_back(x)

        return forward


def _cast(dtype):

    @tf.function
    @nest_map
    def cast_fn(x):
        return tf.cast(x, dtype)

    return cast_fn


class RK4Solver(FixGridODESolver):

    def __init__(self, dt, **kwargs):
        super().__init__(rk4_step_fn, dt, **kwargs)


def rk4_step_fn(f):

    @tf.function
    def step_forward(t, dt, x):
        k1 = f(t, x)

        @nest_map
        def g(x, k1):
            return x + dt * k1 / 2

        k2 = f(t + dt / 2, g(x, k1))
        k3 = f(t + dt / 2, g(x, k2))

        @nest_map
        def h(x, k3):
            return x + dt * k3

        k4 = f(t + dt, h(x, k3))

        @nest_map
        def k(x, k1, k2, k3, k4):
            return x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

        return k(x, k1, k2, k3, k4)

    return step_forward
