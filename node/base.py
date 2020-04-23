"""Definitions"""

import tensorflow as tf


class TimeMeta(type):

  def __instancecheck__(self, instance):
    """For overloading `isinstance`."""
    if not tf.is_tensor(instance):
      return False
    if not _is_scalar(instance):
      return False
    if not instance.dtype.is_floating:
      return False
    return True


def _is_scalar(x: tf.Tensor) -> bool:
  shape = x.get_shape()
  return shape.ndims == 0


def _is_float(x: tf.Tensor) -> bool:
  return x.dtype.is_floating


class Time(metaclass=TimeMeta):
  """Type that represents `tf.Tensor` with shape `[]` and float dtype."""


class PhasePointMeta(type):

  def __instancecheck__(self, instance):
    """For overloading `isinstance`."""
    for x in tf.nest.flatten(instance):
      if not tf.is_tensor(x):
        return False
      if not x.dtype.is_floating and not x.dtype.is_complex:
        return False
    return True


class PhasePoint(metaclass=PhasePointMeta):
  """Type that represents `tf.Tensor` or nest structure of `tf.Tensor`."""


class PhaseVectorField:
  """Type that represents callable with inputs `Time` and `PhasePoint`
  and output `PhasePoint`. The input and output phase points share the same
  shape and dtype."""

  @tf.function
  def __call__(self, t: Time, x: PhasePoint) -> PhasePoint:

    # check inputs
    assert isinstance(t, Time)
    assert isinstance(x, PhasePoint)

    # compute output
    y = self.call(t, x)

    # check output
    assert isinstance(y, PhasePoint)
    tf.nest.assert_same_structure(x, y)

    return y

  def call(self, t: Time, x: PhasePoint) -> PhasePoint:
    return NotImplemented


def phase_vector_field(fn):
  """Decorator that converts a function to phase vector field."""

  def call_method(t, x):
    return fn(t, x)

  pvf = PhaseVectorField()
  setattr(pvf, 'call', fn)
  return pvf


class ODESolver:
  r"""
  ```math

  $$ \text{ode_solver}(f, t_0, t_N, z(t_0)) := z(t_0) + \int_{t_0}^{t_N} f(z(t), t) dt $$  # noqa:E501

  which is exectly the $z(t_N)$.

  ```
  """

  def __call__(self, phase_vector_field):
    """Returns a function that pushes the initial phase point to the final
    along the phase vector field.

    Why So Strange:
      This somehow strange signature is for TF's efficiency.
      For TF>=2, it compiles python code to graph just in time,
      demanding that all the arguments and outputs are `tf.Tensor`s
      or lists of `tf.Tensor`s, and no function.

    Args:
      phase_vector_field: PhaseVectorField

    Returns: Callable[[Time, Time, PhasePoint], PhasePoint]
      Args:
        start_time: Time
        end_time: Time
        initial_phase_point: PhasePoint
      Returns: PhasePoint
    """
    return NotImplemented


class DynamicalODESolver:

  def __call__(self, phase_vector_field, stop_condition):
    """
    Args:
      phase_vector_field: PhaseVectorField
      stop_condition: Callable[[Time, PhasePoint], bool]

    Returns: Callable[[Time, PhasePoint], PhasePoint]
      Args:
        start_time: Time
        initial_phase_point: PhasePoint
      Returns: Tuple[Time, PhasePoint]
        The end time and the final phase point.
    """
    return NotImplemented
