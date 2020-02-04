"""
Reference:
https://github.com/kmkolasinski/deep-learning-notes/blob/master/seminars/2019-03-Neural-Ordinary-Differential-Equations/1A.Adjoint_method.ipynb  # noqa:E501
"""


import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
from node.solvers import (
    FixGridODESolver, FixGridODESolverWithTrajectory, rk4_step_fn)
from node.core import reverse_mode_derivative


t0 = tf.constant(0.)
t1 = tf.constant(25.)
data_size = 200
h0 = tf.constant([[1., 0.]])
W = tf.constant([[-0.1, 1.0], [-0.2, -0.1]])


@tf.function
def f(t, h):
    return tf.matmul(h, W)


ode_solver = FixGridODESolver(rk4_step_fn, data_size)
ode_solver_with_traj = FixGridODESolverWithTrajectory(rk4_step_fn, data_size)
forward = ode_solver(f)
traj_forward = ode_solver_with_traj(f)

hN, states_history = traj_forward(t0, t1, h0)
initial_path = np.concatenate(states_history)


def plot_trajectory(trajectories, fig=True):
    if fig:
        plt.figure(figsize=(5, 5))

    for path in trajectories:
        if type(path) == tuple:
            c, label, path = path
            plt.plot(*path.T, c, lw=2, label=label)
        else:
            plt.plot(*path.T, lw=2)
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()


optimizer = tf.compat.v1.train.MomentumOptimizer(1e-2, momentum=0.95)
forward = ode_solver(f)
h0_var = tf.Variable(h0)
hN_target = tf.constant([[0., 0.5]])
backward = reverse_mode_derivative(ode_solver, f, [])


@tf.function
def compute_gradients_and_update():
    with tf.GradientTape() as g:
        hN = forward(t0, t1, h0_var)
        g.watch(hN)
        loss = tf.reduce_sum((hN_target - hN)**2)

    dLoss = g.gradient(loss, hN)  # same what 2 * (hN_target - hN)
    h0_reconstruction, dfdh0, dWeights = backward(t0, t1, hN, dLoss)
    optimizer.apply_gradients([(dfdh0, h0_var)])
    return loss


loss_history = []
for step in tqdm(range(201)):
    loss = compute_gradients_and_update()
    loss_history.append(loss.numpy())

    if step % 50 == 0:
        yN, states_history_model = traj_forward(t0, t1, h0_var)
        plot_trajectory([
            ("r", "initial", initial_path),
            ("g", "optimized", np.concatenate(states_history_model.numpy()))])
