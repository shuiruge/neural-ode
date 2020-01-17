"""Implements the materials in the "Energy Based" section of the doc."""

import tensorflow as tf
from node.utils.nest import nest_map


class Energy:
    """
    Args:
        linear_trans: Callable[[PhasePoint], PhasePoint]
            Positive defined linear transformation. If `None`, use `identity`.
        static_field: PhaseVectorField
            Shall be static.
    """

    def __init__(self, linear_trans, static_field):
        self.linear_trans = linear_trans
        self.static_field = static_field

    @tf.function
    def __call__(self, x):
        """
        Args:
            x: PhasePoint

        Returns: tf.Tensor
            Shape `[batch_size]`. The energy per sample.
        """
        f = self.static_field(tf.constant(0.), x)  # the `t`-arg is arbitrary.

        rank = len(x.shape)
        sum_axes = list(range(1, rank))  # excludes the batch-axis.

        return 0.5 * tf.reduce_sum(self.linear_trans(f) * f, sum_axes)


def identity(x):
    """The identity linear transform.

    Args:
        x: PhasePoint

    Returns: PhasePoint
    """
    return nest_map(tf.identity, x)


def energy_based(linear_trans_1, linear_trans_2, static_field):
    """
    Args:
        linear_trans_1: Callable[[PhasePoint], PhasePoint]
            Positive defined linear transformation. The $U$ transform.
        linear_trans_2: Callable[[PhasePoint], PhasePoint]
            Positive defined linear transformation. The $W$ transform.
        static_field: PhaseVectorField
            Shall be static.

    Returns: PhaseVectorField
        Static.
    """

    @tf.function
    def energy_based_static_field(_, x):
        with tf.GradientTape() as g:
            g.watch(x)
            f = static_field(_, x)
        vjp = g.gradient(f, x, linear_trans_2(f),
                         unconnected_gradients='zero')
        return -linear_trans_1(vjp)

    return energy_based_static_field
