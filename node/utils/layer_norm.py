import tensorflow as tf
from node.utils.nest import nest_map


@nest_map
@tf.custom_gradient
def layer_norm(x, axes=[-1], eps=1e-8):
  r"""Has property

  \partial f^a / \partial x^b = \partial f^b / \partial x^a

  """
  mean, variance = tf.nn.moments(x, axes, keepdims=True)
  std = tf.sqrt(variance) + eps
  y = (x - mean) / std

  def _grad_fn(grad_y):
    return 1 / std * (
        grad_y - 1 -
        tf.reduce_mean(grad_y * y, axis=axes, keepdims=True) * y)

  def grad_fn(*grad_ys, **kwargs):
    return [_grad_fn(_) for _ in grad_ys]

  return y, grad_fn

