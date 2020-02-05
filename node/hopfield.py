"""Implements the materials in the "Energy Based" section of the doc."""

import tensorflow as tf
from node.utils.nest import nest_map


def check_lower_bound(lower_bound):
  """
  Args:
    lower_bound: float
  
  Returns:
    Decorator that checks the lower bound of the outputs of the decorated
    function.
  """
  with tf.name_scope('lower_bound'):
    lower_bound = tf.convert_to_tensor(lower_bound)

  @nest_map
  def _check_lower_bound(y):
    tf.debugging.assert_greater_equal(y, lower_bound)

  def decorator(fn):

    @tf.function
    def lower_bounded_fn(*args, **kwargs):
      y = fn(*args, **kwargs)
      with tf.name_scope('check_lower_bound'):
        _check_lower_bound(y)
      return y
    
    return lower_bounded_fn
  
  return decorator


@nest_map
def identity(x, name='identity'):
  """The identity linear transform.

  Args:
    x: PhasePoint

  Returns: PhasePoint
  """
  with tf.name_scope(name):
    return x


def rescale(factor):
  """
  The `factor` shall be positive, for being postive defined.

  Args:
    factor: float

  Returns: Callable[[PhasePoint], PhasePoint]
  """
  with tf.name_scope('rescale_factor'):
    assert factor > 0.
    factor = tf.convert_to_tensor(factor)

  @tf.function
  @nest_map
  def rescale_fn(x, name='rescale'):
    with tf.name_scope(name):
      return factor * x

  return rescale_fn


def hopfield(linear_transform, lower_bounded_fn):
  r"""Returns a static phase vector field defined by

  ```
  \begin{equation}
    \frac{dx^{\alpha}}{dt} (t) = - U^{\alpha \beta} \
      \frac{\partial E}{\partial x^\beta} \left( x(t) \right),
  \end{equation}

  where $U$ is a positive defined linear transformation, and $E$ a lower
  bounded function.
  ```

  Args:
    linear_trans: Callable[[PhasePoint], PhasePoint]
      Positive defined linear transformation. The $U$ transform.
    lower_bounded_fn: LowerBoundedFunction

  Returns: PhaseVectorField
  """

  @tf.function
  def static_field(_, x, name='hopfield_field'):
    with tf.name_scope(name):
      with tf.GradientTape() as g:
        g.watch(x)
        e = lower_bounded_fn(x)
      grad = g.gradient(e, x, unconnected_gradients='zero')
      return -linear_transform(grad)

  return static_field
