import tensorflow as tf
from typing import Callable


PhasePoint = tf.Tensor
Time = float
PhaseVectorField = Callable[[PhasePoint, Time], PhasePoint]


def fix_grid_odeint(step_fn):
    r"""ODE integration with fix grid methods.
    
    C.f. the slides of NeuralODE in the deep-learning-notes project
    (GitHub @kmkolasinski).

    The interface follows that in scipy.integrate.

    Args:
        step_fn: Callable[[PhaseVectorField, PhasePoint, Time, Time],
                          PhasePoint]
            Step-function for fixed grid integration method, e.g. Euler's
            method. The last two arguments represents time and time-
            difference respectively.
    
    Returns: Callable[[PhaseVectorField, PhasePoint, List[Time]],
                      PhasePoint]
    """

    @tf.function
    def odeint(func, initial_value, interval):
        """
        Args:
            func: PhaseVectorField
            initial_value: tensor-like
            interval: List[Time]

        Returns: tf.Tensor
            The same shape and dtype as the `initial_value`.
        """
        n_grids = len(interval)
        phase_point = initial_value
        for i in range(n_grids - 1):
            time = interval[i]
            time_diff = interval[i + 1] - interval[i]
            phase_point = step_fn(func, phase_point, time, time_diff)
        return phase_point

    return odeint


@tf.function
def euler_step_fn(f, x, t, dt):
    return x + dt * f(x, t)


@tf.function
def rk2_step_fn(f, x, t, dt):
    k1 = f(x, t)
    k2 = f(x + k1 * dt, t + dt)
    return x + (k1 + k2) / 2 * dt


@tf.function
def rk4_step_fn(f, x, t, dt):
    k1 = f(x, t)
    k2 = f(x + k1 * dt / 2, t + dt / 2)
    k3 = f(x + k2 * dt / 2, t + dt / 2)
    k4 = f(x + k3 * dt, t + dt)
    return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6 * dt


if __name__ == '__main__':

    odeint = fix_grid_odeint(rk2_step_fn)

    # Example 1

    def f(x, t):
        u, v = tf.unstack(x)
        du_dt = v
        dv_dt = 5 * v - 6 * u
        return tf.stack([du_dt, dv_dt])


    x0 = [1., 1.]
    ts = tf.range(0, 1, 1e-3)
    x1 = odeint(f, x0, ts)
    print(x1)

    # Example 2

    def f(x, t):
        dx_dt = tf.sin(t ** 2) * x
        return dx_dt

    x0 = [1.]
    ts = tf.range(0, 8, 1e-2)
    x1 = odeint(f, x0, ts)
    print(x1)
