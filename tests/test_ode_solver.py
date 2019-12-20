import tensorflow as tf
from node.fix_grid import RKSolver

solver = RKSolver(0.01)


# Example 1


@tf.function
def f(t, x):
    u, v = tf.unstack(x)
    du_dt = v
    dv_dt = 5 * v - 6 * u
    return tf.stack([du_dt, dv_dt])


x0 = tf.constant([1., 1.])
forward = solver(f)
x1 = forward(0., 1., x0)
print(x1)


# Example 2


@tf.function
def f(t, x):
    dx_dt = tf.sin(t ** 2) * x
    return dx_dt


x0 = tf.constant([1.])
forward = solver(f)
x1 = forward(0., 1., x0)
print(x1)
