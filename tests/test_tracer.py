"""
Reference:
https://github.com/kmkolasinski/deep-learning-notes/blob/master/seminars/2019-03-Neural-Ordinary-Differential-Equations/1.Demo_spiral.ipynb  # noqa:E501
"""

import tensorflow as tf
import matplotlib.pyplot as plt
from node.base import tracer
from node.fix_grid import RKSolver


data_size = 1000
n_iters = 3000
batch_size = 16
true_y0 = tf.constant([[2., 0]])
true_A = tf.constant([[-0.1, 2.0], [-2.0, -0.1]])


@tf.function
def f(t, x):
    return tf.matmul(x ** 3, true_A)


t0 = tf.constant(0.)
t1 = tf.constant(25.)
dt = (t1 - t0) / data_size
true_y0 = tf.constant(true_y0)

trace = tracer(RKSolver(dt * 0.2), f)
trajectory = trace(t0, t1, dt, true_y0)
trajectory = trajectory.numpy().reshape([-1, 2])

plt.plot([x for x, y in trajectory],
         [y for x, y in trajectory])
plt.show()
