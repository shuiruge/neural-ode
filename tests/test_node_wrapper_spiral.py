"""
This script tests

    * `node.base.get_node_function`, and that

    * the RAM occupaton is O(1).

Reference:
https://github.com/kmkolasinski/deep-learning-notes/blob/master/seminars/2019-03-Neural-Ordinary-Differential-Equations/1.Demo_spiral.ipynb  # noqa:E501
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from node.core import get_node_function, tracer
from node.fix_grid import RKSolver


data_size = 1000
batch_time = 20  # this seems to works the best ...
# batch_time = 200  # comparing for varifying that memory cost is O(1)
n_iters = 3000
batch_size = 16
true_y0 = tf.constant([[2., 0]])
true_A = tf.constant([[-0.1, 2.0], [-2.0, -0.1]])


@tf.function
def f(t, x):
    return tf.matmul(x ** 3, true_A)


t0 = tf.constant(0.)
t1 = tf.constant(25.)
dt = (t1 - t0) / (data_size - 1)
true_y0 = tf.constant(true_y0)

traj_forward = tracer(RKSolver(1e-2), f)
true_y = traj_forward(t0, t1, dt, true_y0)
true_y = true_y.numpy().reshape([data_size, 2])
ts = tf.linspace(t0, t1, data_size).numpy()


def plot_spiral(trajectory):
    plt.plot([x for x, y in trajectory],
             [y for x, y in trajectory])


model = tf.keras.Sequential([
    tf.keras.layers.Dense(50, activation="tanh"),
    tf.keras.layers.Dense(2)])
model.build([None, 2])
var_list = model.trainable_variables


@tf.function
def network(t, x):
    h = x ** 3
    return model(h)


node_network = get_node_function(RKSolver(dt), t0, network)


def get_batch():
    """Returns initial point and last point over sampled frament of
    trajectory"""
    starts = np.random.choice(
        np.arange(data_size - batch_time - 1, dtype=np.int64),
        batch_size,
        replace=False)
    ends = starts + batch_time
    batch_y0 = true_y[starts]  # (batch_size, 2) -> initial point
    batch_yN = true_y[ends]
    return tf.constant(batch_y0), tf.constant(batch_yN)


optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=1e-4)
tb = ts[batch_time]


@tf.function
def compute_gradients_and_update(batch_y0, batch_yN):
    """Takes start positions (x0, y0) and final positions (xN, yN)"""
    with tf.GradientTape() as g:
        pred_y = node_network(tb, batch_y0)
        loss = tf.reduce_mean(tf.abs(pred_y - batch_yN))
    grads = g.gradient(loss, var_list)
    optimizer.apply_gradients(zip(grads, var_list))
    return loss


loss_history = []
for step in range(n_iters):
    loss = compute_gradients_and_update(*get_batch())
    loss_history.append(loss.numpy())
    print(f'{step} - {loss.numpy()}')

    if step % 500 == 0:
        states_history_model = (traj_forward(t0, t1, dt, true_y0)
                                .numpy()
                                .reshape([data_size, 2]))
        plot_spiral(true_y)
        plot_spiral(states_history_model)
        plt.show()
