"""Implements the materials in the "Continuum of Hopfield" section of the
documentation."""

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
  lower_bound = tf.convert_to_tensor(lower_bound)

  @nest_map
  def _check_lower_bound(y):
    tf.debugging.assert_greater_equal(y, lower_bound)

  def decorator(fn):

    @tf.function
    def lower_bounded_fn(*args, **kwargs):
      y = fn(*args, **kwargs)
      _check_lower_bound(y)
      return y

    return lower_bounded_fn

  return decorator


@nest_map
def identity(x):
  """The identity linear transform.

  Args:
    x: PhasePoint

  Returns: PhasePoint
  """
  return x


def rescale(factor):
  """
  The `factor` shall be positive, for being postive defined.

  Args:
    factor: float

  Returns: Callable[[PhasePoint], PhasePoint]
  """
  assert factor > 0.
  factor = tf.convert_to_tensor(factor)

  @tf.function
  @nest_map
  def rescale_fn(x):
    return factor * x

  return rescale_fn


def hopfield(energy, linear_transform=identity):
  r"""Returns a static phase vector field defined by

  ```
  \begin{equation}
    \frac{dx^a}{dt} (t) = - U^{a b} \
      \frac{\partial \mathcal{E}}{\partial x^b} \left( x(t) \right),
  \end{equation}

  where $U$ is a positive defined linear transformation, and energy
  $\mathcal{E}$ a lower bounded scalar function.
  ```

  Args:
    energy: Callable[[PhasePoint], tf.Tensor]
      The lower bounded scalar (per sample) function `\mathcal{E}`.

      Per sample means that it produces scalar for each sample in a batch of
      inputs. Say, if the input shape is `[batch_size, model_dim_1, ...]`, then
      the output shape will be `[batch_size]`.

    linear_transform: Callable[[PhasePoint], PhasePoint]
      Positive defined linear transformation. The $U$ transform.

  Returns: PhaseVectorField
  """

  @tf.function
  def static_field(_, x):
    with tf.GradientTape() as g:
      g.watch(x)
      e = energy(x)
    energy_gradient = g.gradient(e, x, unconnected_gradients='zero')
    return -linear_transform(energy_gradient)

  return static_field



def get_stop_condition(pvf, max_delta_t, tolerance):
  max_delta_t = tf.convert_to_tensor(float(max_delta_t))
  tolerance = tf.convert_to_tensor(float(tolerance))

  @tf.function
  def stop_condition(t0, x0, t, x):
    if tf.abs(t - t0) > max_delta_t:
      return True
    max_abs_velocity = tf.reduce_max(tf.abs(pvf(t, x)))
    if max_abs_velocity < tolerance:
      return True
    return False

  return stop_condition
