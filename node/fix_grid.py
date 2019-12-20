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
        num_grids: int

    Returns: ODESolver
    """

    def __init__(self, step_fn, num_grids):
        self.step_fn = step_fn
        self.num_grids = num_grids

    def __call__(self, phase_vector_field):

        @tf.function
        def forward(start_time, end_time, initial_phase_point):
            x, t = initial_phase_point, start_time
            interval = tf.linspace(start_time, end_time, self.num_grids)
            for next_t in interval[1:]:
                dt = next_t - t
                x = self.step_fn(phase_vector_field, t, dt, x)
                t = next_t
            return x

        return forward


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

    if list_input:
        new_x = [xi + k1i * dt / 2 for xi, k1i in zip(x, k1)]
    else:
        new_x = x + k1 * dt / 2
    new_t = t + dt / 2
    k2 = f(new_t, new_x)

    if list_input:
        new_x = [xi + k2i * dt / 2 for xi, k2i in zip(x, k2)]
    else:
        new_x = x + k2 * dt / 2
    new_t = t + dt / 2
    k3 = f(new_t, new_x)

    if list_input:
        new_x = [xi + k3i * dt for xi, k3i in zip(x, k3)]
    else:
        new_x = x + k3 * dt
    new_t = t + dt
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
            x, t, i = initial_phase_point, start_time, 0
            xs = tf.TensorArray(tf.float32, self.num_grids).write(i, x)

            interval = tf.linspace(start_time, end_time, self.num_grids)
            for next_t in interval[1:]:
                dt = next_t - t
                x = self.step_fn(phase_vector_field, t, dt, x)
                t, i = next_t, i + 1
                xs = xs.write(i, x)
            return x, xs.stack()

        return forward
