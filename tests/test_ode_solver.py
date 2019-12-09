import tensorflow as tf
from node.ode_solver import FixGridODESolver, rk4_step_fn

ode_solver = FixGridODESolver(rk4_step_fn, 100)


# Example 1


def f(x, t):
    u, v = tf.unstack(x)
    du_dt = v
    dv_dt = 5 * v - 6 * u
    return tf.stack([du_dt, dv_dt])


x0 = tf.constant([1., 1.])
x1 = ode_solver(f, 0, 1, x0)
print(x1)


# Example 2


def f(x, t):
    dx_dt = tf.sin(t ** 2) * x
    return dx_dt


x0 = tf.constant([1.])
x1 = ode_solver(f, 0, 8, x0)
print(x1)
