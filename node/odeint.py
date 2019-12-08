import tensorflow as tf
from typing import List, Callable


PhasePoint = List[tf.Tensor]
Time = float
PhaseVectorField = Callable[[PhasePoint, Time], PhasePoint]


def fix_grid_odeint(step_fn):
    r"""ODE integration with fix grid methods.
    
    C.f. the slides of NeuralODE in the deep-learning-notes project
    (GitHub @kmkolasinski).

    The interface follows that in scipy.integrate.

    Args:
        step_fn: Callable[[PhaseVectorField, Time, Time, PhasePoint],
                          PhasePoint]
            Step-function for fixed grid integration method, e.g. Euler's
            method. The last two arguments represents time and time-
            difference respectively.
    
    Returns: Callable[[PhaseVectorField, List[Time], PhasePoint],
                      PhasePoint]
    """

    @tf.function
    def odeint(func, interval, initial_value):
        """
        Args:
            func: PhaseVectorField
            interval: List[Time]
            initial_value: PhasePoint

        Returns: PhasePoint
        """
        n_grids = len(interval)
        phase_point = initial_value
        for i in range(n_grids - 1):
            time = interval[i]
            time_diff = interval[i + 1] - interval[i]
            phase_point = step_fn(func, time, time_diff, phase_point)
        return phase_point

    return odeint


def apply_to_maybe_list(step_fn_comp):

    def step_fn(f, t, dt, x):
        if not isinstance(x, list):
            return step_fn_comp(f, t, dt, x)
        else:
            return [step_fn_comp(f, t, dt, xi) for xi in x]

    return step_fn


@apply_to_maybe_list
def euler_step_fn(f, t, dt, x):
    return x + dt * f(x, t)


@apply_to_maybe_list
def rk2_step_fn(f, t, dt, x):
    k1 = f(x, t)
    k2 = f(x + k1 * dt, t + dt)
    return x + (k1 + k2) / 2 * dt


@apply_to_maybe_list
def rk4_step_fn(f, t, dt, x):
    k1 = f(x, t)
    k2 = f(x + k1 * dt / 2, t + dt / 2)
    k3 = f(x + k2 * dt / 2, t + dt / 2)
    k4 = f(x + k3 * dt, t + dt)
    return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6 * dt
