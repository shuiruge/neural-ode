import tensorflow as tf
from node.base import ODESolver
from node.core import nest_map


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
        diff_time: Optional[float]

    Returns: ODESolver
    """

    def __init__(self, step_fn, diff_time):
        self.step_fn = step_fn
        self.diff_time = diff_time

    def __call__(self, phase_vector_field):
        step_forward = self.step_fn(phase_vector_field)

        @tf.function
        def forward(start_time, end_time, initial_phase_point):
            t0, tN, x = start_time, end_time, initial_phase_point
            dt = self.diff_time

            if abs(tN - t0) < dt:
                return step_forward(t0, dt, x)

            # TODO: illustrate why the `dt` shall be computed as so
            dt = tf.where(tN > t0, dt, -dt)

            # To compute the `ts`, it's better to use `tf.linspace` instead
            # of using `tf.range`, since `tf.range(t0, tN, dt)` will raise
            # error when `t0 > tN`, which will happen in the backward process.
            # However, `tf.linspace` works well in this case.
            N = int((tN - t0) / dt + 1)
            ts = tf.linspace(t0, tN, N)

            for t in ts:
                x = step_forward(t, dt, x)
            return x

        return forward


class RKSolver(FixGridODESolver):

    def __init__(self, diff_time):
        super().__init__(rk4_step_fn, diff_time)


def euler_step_fn(f):

    @tf.function
    def step_forward(t, dt, x):
        k1 = f(t, x)
        new_x = nest_map(lambda x, k1: x + k1 * dt, x, k1)
        return new_x

    return step_forward


def rk4_step_fn(f):

    @tf.function
    def step_forward(t, dt, x):
        k1 = f(t, x)

        new_t = t + dt / 2
        new_x = nest_map(lambda x, k1: x + k1 * dt / 2, x, k1)
        k2 = f(new_t, new_x)

        new_t = t + dt / 2
        new_x = nest_map(lambda x, k2: x + k2 * dt / 2, x, k2)
        k3 = f(new_t, new_x)

        new_t = t + dt
        new_x = nest_map(lambda x, k3: x + k3 * dt, x, k3)
        k4 = f(new_t, new_x)

        new_x = nest_map(
            lambda x, k1, k2, k3, k4:
                x + (k1 + 2 * k2 + 2 * k3 + k4) / 6 * dt,
            x, k1, k2, k3, k4)
        return new_x

    return step_forward
