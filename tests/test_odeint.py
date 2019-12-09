import numpy as np
import tensorflow as tf
from node.odeint import fix_grid_odeint, rk2_step_fn

odeint = fix_grid_odeint(rk2_step_fn)

# Example 1

def f(x, t):
    u, v = tf.unstack(x)
    du_dt = v
    dv_dt = 5 * v - 6 * u
    return tf.stack([du_dt, dv_dt])


x0 = tf.constant([1., 1.])
ts = tf.range(0, 1, 1e-2)
x1 = odeint(f, ts, x0)
print(x1)

# Example 2

def f(x, t):
    dx_dt = tf.sin(t ** 2) * x
    return dx_dt

x0 = tf.constant([1.])
ts = tf.range(0, 8, 1e-2)
x1 = odeint(f, ts, x0)
print(x1)
