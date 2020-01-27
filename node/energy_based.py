"""Implements the materials in the "Energy Based" section of the doc."""

import tensorflow as tf
from node.utils.nest import nest_map


class LowerBoundedFunction:

    def __init__(self, check_lower_bound=False):
        self.check_lower_bound = check_lower_bound

    @tf.function
    def __call__(self, *args, **kwargs):
        y = self.call(*args, **kwargs)  # returns scalar.
        if self.check_lower_bound:
            infimum = tf.convert_to_tensor(self.infimum)
            tf.debugging.assert_greater_equal(y, infimum)
        return y

    @property
    def infimum(self):
        """Scalar-like"""
        return NotImplemented

    def call(self, *args, **kwargs):
        """Inputs `tf.Tensor` or nested `tf.Tensor`s (as the `*args`),
        and returns a scalar `tf.Tensor`."""
        return NotImplemented


class Energy(LowerBoundedFunction):
    """
    Args:
        linear_trans: Callable[[PhasePoint], PhasePoint]
            Positive defined linear transformation.
        static_field: PhaseVectorField
            Shall be static.
    """

    def __init__(self, linear_trans, static_field, **kwargs):
        super().__init__(**kwargs)

        self.linear_trans = linear_trans
        self.static_field = static_field

    @property
    def infimum(self):
        """Scalar-like"""
        return 0.

    @tf.function
    def call(self, x):
        """
        Args:
            x: PhasePoint

        Returns: tf.Tensor
            Shape `[batch_size]`. The energy per sample.
        """
        arbitrary_time = tf.convert_to_tensor(0.)
        y = self.static_field(arbitrary_time, x)

        rank = len(x.shape)
        sum_axes = list(range(1, rank))  # excludes the batch-axis.

        return 0.5 * tf.reduce_sum(self.linear_trans(y) * y, sum_axes)


def identity(x):
    """The identity linear transform.

    Args:
        x: PhasePoint

    Returns: PhasePoint
    """
    return nest_map(tf.identity, x)


def rescale(factor):
    """
    The `factor` shall be positive, for being postive defined.

    Args:
        factor: float

    Returns: Callable[[PhasePoint], PhasePoint]
    """
    assert factor > 0.
    factor = tf.convert_to_tensor(factor)

    def _rescale_fn(x):
        return factor * x

    @tf.function
    def rescale_fn(x):
        return nest_map(_rescale_fn, x)

    return rescale_fn


def energy_based(linear_transform, lower_bounded_fn):
    r"""Returns a static phase vector field defined by

    ```
    \begin{equation}
        \frac{dx^{\alpha}}{dt} (t) = - U^{\alpha \beta}
                                       \frac{\partial E}{\partial x^\beta}
                                       \left( z(t) \right),
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
    def static_field(_, x):
        with tf.GradientTape() as g:
            g.watch(x)
            e = lower_bounded_fn(x)
        grad = g.gradient(e, x, unconnected_gradients='zero')
        return -linear_transform(grad)

    return static_field
