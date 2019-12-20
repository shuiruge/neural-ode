import tensorflow as tf
from node.base import ODESolver


class FixGridODESolver(ODESolver):
    r"""ODE solver with fix grid methods.

    C.f. the slides of Neural ODE in the deep-learning-notes project
    (GitHub @kmkolasinski).

    Args:
        step_fn: Callable[[PhaseVectorField, Time, Time, PhasePoint],
                          PhasePoint]
            Step-function for fixed grid integration method, e.g. Euler's
            method. The last two arguments represents time and time-
            difference respectively.
        diff_time: Optional[float]
        num_grids: Optional[int]

    Returns: ODESolver
    """

    def __init__(self, step_fn, diff_time):
        self.step_fn = step_fn
        self.diff_time = diff_time

    def __call__(self, phase_vector_field):

        @tf.function
        def forward(start_time, end_time, initial_phase_point):
            # TODO: illustrate why the `dt` shall be computed as so
            dt = tf.where(end_time > start_time,
                          self.diff_time,
                          -self.diff_time)

            # To compute the `ts`, it's better to use `tf.linspace` instead
            # of using `tf.range`, since `tf.range(start_time, end_time, dt)`
            # will raise error when `start_time > end_time`, which will happen
            # in the backward process. However, `tf.linspace` works well in
            # this case.
            num_grids = int((end_time - start_time) / dt + 1)
            ts = tf.linspace(start_time, end_time, num_grids)

            x = initial_phase_point
            for t in ts:
                x = self.step_fn(phase_vector_field, t, dt, x)
            return x

        return forward


class RKSolver(FixGridODESolver):

    def __init__(self, diff_time=None):
        super().__init__(rk4_step_fn, diff_time)


@tf.function
def euler_step_fn(f, t, dt, x):
    list_input = isinstance(x, list)

    k1 = f(t, x)

    if list_input:
        new_x = [xi + k1i * dt for xi, k1i in zip(x, k1)]
    else:
        new_x = x + k1 * dt
    return new_x


@tf.function
def rk4_step_fn(f, t, dt, x):
    list_input = isinstance(x, list)

    k1 = f(t, x)

    new_t = t + dt / 2
    if list_input:
        new_x = [xi + k1i * dt / 2 for xi, k1i in zip(x, k1)]
    else:
        new_x = x + k1 * dt / 2
    k2 = f(new_t, new_x)

    new_t = t + dt / 2
    if list_input:
        new_x = [xi + k2i * dt / 2 for xi, k2i in zip(x, k2)]
    else:
        new_x = x + k2 * dt / 2
    k3 = f(new_t, new_x)

    new_t = t + dt
    if list_input:
        new_x = [xi + k3i * dt for xi, k3i in zip(x, k3)]
    else:
        new_x = x + k3 * dt
    k4 = f(new_t, new_x)

    if list_input:
        new_x = [xi + (k1i + 2 * k2i + 2 * k3i + k4i) / 6 * dt
                 for xi, k1i, k2i, k3i, k4i in zip(x, k1, k2, k3, k4)]
    else:
        new_x = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6 * dt
    return new_x


class FixGridODESolverWithTrajectory:
    """For test"""

    def __init__(self, step_fn, num_grids):
        self.step_fn = step_fn
        self.num_grids = num_grids

    def __call__(self, phase_vector_field):

        @tf.function
        def forward(start_time, end_time, initial_phase_point):
            interval = tf.linspace(start_time, end_time, self.num_grids)
            i, t, x = 0, start_time, initial_phase_point
            xs = tf.TensorArray(tf.float32, self.num_grids).write(i, x)

            for next_t in interval[1:]:
                dt = next_t - t
                x = self.step_fn(phase_vector_field, t, dt, x)
                i, t = i + 1, next_t
                xs = xs.write(i, x)
            return x, xs.stack()

        return forward
