import tensorflow as tf
from node.wrapper import node_wrapper
from node.solvers import RK4Solver


x1 = tf.constant([[2., 0]])
x2 = tf.constant([[-0.1, 2.0], [-2.0, -0.1]])

solver = RK4Solver(0.1)
dense = tf.keras.layers.Dense(2)


@node_wrapper(solver, tf.constant(0.))
@tf.function
def f(t, x):
  x0, x1 = tf.unstack(x[0], axis=-1)
  y0 = x0 + 2 * x1
  y1 = -3 * x0 + x1
  return [dense(tf.stack([y0, y1], axis=-1)), tf.zeros_like(x[1])]


with tf.GradientTape() as g:
  g.watch(x1)
  y = f(tf.constant(1.), [x1, x2])
print(g.gradient(y, x1))

with tf.GradientTape() as g:
  g.watch(x2)
  y = f(tf.constant(1.), [x1, x2])
print(g.gradient(y, x2))
