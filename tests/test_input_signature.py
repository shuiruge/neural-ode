import tensorflow as tf
from node.core import get_node_function
from node.solvers.runge_kutta import RK4Solver


tf.random.set_seed(42)
dense = tf.keras.layers.Dense(10)


def f(t, x):
    return dense(x)


solver = RK4Solver(0.1)
t0 = tf.constant(0.)
signature = [tf.TensorSpec(shape=[None, 3], dtype=tf.float32)]
node_f = get_node_function(solver, t0, f, signature=signature)
t1 = tf.constant(1.)


def test_node_f(x0):
    with tf.GradientTape() as g:
        g.watch(x0)
        x1 = node_f(t1, x0)
    grad = g.gradient(x1, x0)
    print(grad)

    with tf.GradientTape() as g:
        g.watch(x0)
        x1 = node_f(t1, x0)
    grad = g.gradient(x1, x0)
    print(grad)


x0 = tf.random.uniform(shape=[2, 3])
test_node_f(x0)

x0 = tf.random.uniform(shape=[3, 3])
test_node_f(x0)
