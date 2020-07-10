import tensorflow as tf
from node.solvers.dynamical_runge_kutta import (
  DynamicalRK4Solver, DynamicalRKF56Solver)


def test_rk4_solver():
  solver = DynamicalRK4Solver(0.01)

  # the first example

  @tf.function
  def f(t, x):
    u, v = tf.unstack(x)
    du_dt = v
    dv_dt = 5 * v - 6 * u
    return tf.stack([du_dt, dv_dt])

  @tf.function
  def stop_condition(t0, x0, t1, x1):
    return t1 > 1

  x0 = tf.constant([1., 1.])
  forward = solver(f, stop_condition)
  t0 = tf.constant(0.)
  t1, x1 = forward(t0, x0)
  print(t1, x1)

  # the second example

  @tf.function
  def f(t, x):
    dx_dt = tf.sin(t ** 2) * x
    return dx_dt

  @tf.function
  def stop_condition(t0, x0, t1, x1):
    return t1 < 0

  x0 = tf.constant([1.])
  forward = solver(f, stop_condition)
  t0 = tf.constant(1.)
  t1, x1 = forward(t0, x0, reverse=True)
  print(t1, x1)


def test_rkf56_solver():
  solver = DynamicalRKF56Solver(0.01, tol=1e-3, min_dt=1e-2)

  # the first example

  @tf.function
  def f(t, x):
    u, v = tf.unstack(x)
    du_dt = v
    dv_dt = 5 * v - 6 * u
    return tf.stack([du_dt, dv_dt])

  @tf.function
  def stop_condition(t0, x0, t1, x1):
    return t1 > 1

  x0 = tf.constant([1., 1.])
  forward = solver(f, stop_condition)
  t0 = tf.constant(0.)
  t1, x1 = forward(t0, x0)
  print(t1, x1)
  print(solver.diagnostics)

  # the second example

  @tf.function
  def f(t, x):
    dx_dt = tf.sin(t ** 2) * x
    return dx_dt

  @tf.function
  def stop_condition(t0, x0, t1, x1):
    return t1 < 0

  x0 = tf.constant([1.])
  forward = solver(f, stop_condition)
  t0 = tf.constant(1.)
  t1, x1 = forward(t0, x0, reverse=True)
  print(t1, x1)
  print(solver.diagnostics)


print('\ntesting rk4 solver')
test_rk4_solver()
print('succeed in testing rk4 solver\n')

print('\ntesting rkf56 solver')
test_rkf56_solver()
print('succeed in testing rkf56 solver')
