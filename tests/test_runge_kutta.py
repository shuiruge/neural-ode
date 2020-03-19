import tensorflow as tf
from node.solvers.runge_kutta import RK4Solver, RKF56Solver


def test_rk4_solver():
  solver = RK4Solver(0.01)

  # the first example

  @tf.function
  def f(t, x):
    u, v = tf.unstack(x)
    du_dt = v
    dv_dt = 5 * v - 6 * u
    return tf.stack([du_dt, dv_dt])

  x0 = tf.constant([1., 1.])
  forward = solver(f)
  t0 = tf.constant(0.)
  t1 = tf.constant(1.)
  x1 = forward(t0, t1, x0)
  print(x1)

  # the second example

  @tf.function
  def f(t, x):
    dx_dt = tf.sin(t ** 2) * x
    return dx_dt

  x0 = tf.constant([1.])
  forward = solver(f)
  t1 = tf.constant(0.)
  t0 = tf.constant(1.)
  x1 = forward(t0, t1, x0)
  print(x1)


def test_rkf56_solver():
  solver = RKF56Solver(0.1, tol=1e-3)

  # the first example

  @tf.function
  def f(t, x):
    u, v = tf.unstack(x)
    du_dt = v
    dv_dt = 5 * v - 6 * u
    return tf.stack([du_dt, dv_dt])

  x0 = tf.constant([1., 1.])
  forward = solver(f)
  t0 = tf.constant(0.)
  t1 = tf.constant(1.)
  x1 = forward(t0, t1, x0)
  print(x1)

  # the second example

  @tf.function
  def f(t, x):
    dx_dt = tf.sin(t ** 2) * x
    return dx_dt

  x0 = tf.constant([1.])
  forward = solver(f)
  t1 = tf.constant(0.)
  t0 = tf.constant(1.)
  x1 = forward(t0, t1, x0)
  print(x1)


print('\ntesting rk4 solver')
test_rk4_solver()
print('succeed in testing rk4 solver\n')

print('\ntesting rkf56 solver')
test_rkf56_solver()
print('succeed in testing rkf56 solver')
