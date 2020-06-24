import tensorflow as tf
from node.utils.layer_norm import layer_norm


def test_gradients():
  x = tf.random.uniform(shape=[10, 20, 30])

  with tf.GradientTape() as g:
    g.watch(x)
    y = layer_norm(x)
    grad = g.gradient(y, x, x ** 2)
    assert grad.shape == [10, 20, 30]
  print('Succeed in testing gradients.')


def test_variable_gradients():
  x = tf.random.uniform(shape=[10, 20, 30])
  layer = tf.keras.layers.Dense(32)

  with tf.GradientTape() as g:
    g.watch(x)
    z = layer(x)
    y = layer_norm(z)
    grads = g.gradient(y, layer.trainable_variables)
    assert grads[0].shape == [30, 32]
    assert grads[1].shape == [32]
  print('Succeed in testing variable gradients.')


def test_nest_inputs():
  x = [tf.random.uniform(shape=[10, 20, 30]),
       tf.random.uniform(shape=[2, 5])]

  with tf.GradientTape() as g:
    g.watch(x)
    y = layer_norm(x)
    grads = g.gradient(y, x)
  assert grads[0].shape == [10, 20, 30]
  assert grads[1].shape == [2, 5]
  print('Succeed in testing nest inputs.')


if __name__ == '__main__':

  test_gradients()
  test_variable_gradients()
  test_nest_inputs()

