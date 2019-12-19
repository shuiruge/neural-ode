import tensorflow as tf
from node.base import reverse_mode_derivative


# TODO: compare efficiency with the new.
def node_wrapper_old(solver, t0, tN):
    """TODO"""

    def decorator(fn):
        """Decorator that converts an arbitrary function to NODE layer,
        where the original function serves as a phase vector field.

        Args:
            fn: Callable[[tf.Tensor, Time], tf.Tensor]

        Returns: Callable
            Inputs are input and variables. Outputs are the gradients of them.
        """
        forward = solver(fn)

        @tf.custom_gradient
        def node_fn(x):
            """Function that satisfies the signature of `tf.custom_gradient`.

            Args:
                x: tf.Tensor

            Returns:
                The outputs that satisfy the signature of `tf.custom_gradient`.

            References:
                https://tensorflow.google.cn/api_docs/python/tf/custom_gradient
            """
            y = forward(t0, tN, x)

            def grad_fn(grad_y, variables=None):
                backward = reverse_mode_derivative(solver, fn, variables)
                _, grad_by_x, grad_by_vars = backward(t0, tN, y, grad_y)
                return [grad_by_x], grad_by_vars

            return y, grad_fn

        return node_fn

    return decorator


def node_wrapper(solver, t0):
    r"""Returns a decorator which wraps phase vector field `f(x, t)` to
    `F(x, t)` defined as $ F(x, t) = \int_{t_0}^t f(F(x, t), t) dt $,
    where $ F(x, t_0) = x $. That is, the ending phase point at `t` of
    the flow starting at `x` at `t0` on the phase vector field.

    Args:
        solver: ODESolver
        t0: Time
            The start time of the phase flow.

    Returns:
        A decorator.
    """

    def decorator(fn):
        """Decorator that converts an arbitrary function to NODE layer,
        where the original function serves as a phase vector field.

        Args:
            fn: Callable[[tf.Tensor, Time], tf.Tensor]

        Returns: Callable[[tf.Tensor, Time], tf.Tensor]
        """
        forward = solver(fn)

        @tf.function
        def node_fn(x, t):
            """
            Args:
                x: tf.Tensor
                t: Time

            Returns: tf.Tensor
            """

            @tf.custom_gradient
            def custom_gradient_fn(x):
                """For matching the signature of `tf.custom_gradient`
                https://tensorflow.google.cn/api_docs/python/tf/custom_gradient
                """
                y = forward(t0, t, x)

                @tf.function
                def grad_fn(grad_y, variables=None):
                    backward = reverse_mode_derivative(solver, fn, variables)
                    _, grad_by_x, grad_by_vars = backward(t0, t, y, grad_y)
                    return [grad_by_x], grad_by_vars

                return y, grad_fn

            return custom_gradient_fn(x)

        return node_fn

    return decorator
