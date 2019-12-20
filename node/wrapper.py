import tensorflow as tf
from node.base import reverse_mode_derivative


def get_node_function(solver, t0, fn):
    r"""Converts a phase vector field `f(t, x)` to `F(t, x)` which is defined
    as $ F(t, x) = \int_{t_0}^t f(t, F(t, x)) dt $, where $ F(t_0, x) = x $.
    That is, the ending phase point at `t` of the flow starting at `x` at `t0`
    on the phase vector field.

    Args:
        solver: ODESolver
        t0: Time
            The start time of the phase flow.
        fn: Callable[[Time, tf.Tensor], tf.Tensor]

    Returns: Callable[[Time, tf.Tensor], tf.Tensor]
    """
    forward = solver(fn)

    @tf.function
    def node_fn(t, x):
        """
        Args:
            t: Time
            x: tf.Tensor

        Returns: tf.Tensor
        """

        @tf.custom_gradient
        def custom_gradient_fn(x):
            """For matching the signature of `tf.custom_gradient`
            https://tensorflow.google.cn/api_docs/python/tf/custom_gradient
            """
            y = forward(t0, t, x)

            # TF will catch all the variables watched by `tf.GradientTape`,
            # and pass them into the `grad_fn` via the `variables` kwarg.
            @tf.function
            def grad_fn(grad_y, variables=None):
                backward = reverse_mode_derivative(solver, fn, variables)
                _, grad_by_x, grad_by_vars = backward(t0, t, y, grad_y)
                return [grad_by_x], grad_by_vars

            return y, grad_fn

        return custom_gradient_fn(x)

    return node_fn


def node_wrapper(solver, t0):
    """Returns a decorator."""
    return lambda fn: get_node_function(solver, t0, fn)
