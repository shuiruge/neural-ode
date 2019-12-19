import tensorflow as tf
from node.base import reverse_mode_derivative


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

        def node_fn(x, t):
            """TODO"""

            @tf.custom_gradient
            def custom_gradient_fn(x):
                """For matching the signature of `tf.custom_gradient`
                https://tensorflow.google.cn/api_docs/python/tf/custom_gradient
                """
                y = forward(t0, t, x)

                @tf.function  # XXX: essential?
                def grad_fn(grad_y, variables=None):
                    backward = reverse_mode_derivative(solver, fn, variables)
                    _, grad_by_x, grad_by_vars = backward(t0, t, y, grad_y)
                    return [grad_by_x], grad_by_vars

                return y, grad_fn

            return custom_gradient_fn(x)

        return node_fn

    return decorator
