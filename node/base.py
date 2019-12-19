import tensorflow as tf
from typing import Union, List, Callable

Time = float
PhasePoint = Union[tf.Tensor, List[tf.Tensor]]
PhaseVectorField = Callable[[PhasePoint, Time], PhasePoint]


class ODESolver:
    r"""
    Definition
    ----------
    $\text{ode_solver}(f, t_0, t_N, z(t_0)) := z(t_0) + \int_{t_0}^{t_N} f(z(t), t) dt$  # noqa:E501
    which is exectly the $z(t_N)$.
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


def reverse_mode_derivative(ode_solver, network, variables):
    r"""Implements the algorithm 1 in the paper original paper (1806.07366).

    Args:
        ode_solver: ODESolver
        network: Callable[[tf.Tensor, Time], tf.Tensor]
            The $f(x, t)$ in the paper.
        variables: List[tf.Variable]
            The $\theta$ in the paper. In practice, it's a list of variables.
            Thus $\theta = (\theta_1, \ldots)$,

    Returns: Callable for backward propagation
        Args:
            start_time: Time
            end_time: Time
            final_state: tf.Tensor
                The $z^{\alpha}(t_N)$ in the paper. The final outputs of the
                Neural ODE.
            final_loss_gradient: tf.Tensor
                The $\frac{\partial L}{\partial z^{\alpha}(t_N)}$ in the paper.
                The gradient of loss by the final output of the Neural ODE
                (i.e. by the `final_state`).
        Returns: Tuple[tf.Tensor, tf.Tensor, List[tf.Tensor]]
            For the initial state, the gradient of loss by the initial state,
            and the gradient of loss by the variables `variables`. In the
            paper, they are $z(t_0)$, $\partial L / \partial z^{\alpha}(t_0)$,
            and $\partial L / \partial \theta_i^{\alpha}$, respectively.
    """

    @tf.function
    def aug_dynamics(phase_point, time):
        state, adjoint, *_ = phase_point

        with tf.GradientTape() as g:
            g.watch(state)
            output = network(state, time)
        # According to
        # # https://www.tensorflow.org/api_docs/python/tf/custom_gradient
        # `tf.gradients` or `g.gradient`, if the third argument is filled,
        # returns the vector-Jacobian-products directly. In fact, TF
        # implements VJP inside, and compute gradients via VJP.
        grads = g.gradient(output, [state] + variables, -1 * adjoint)

        new_phase_point = [output] + grads
        return new_phase_point

    forward = ode_solver(aug_dynamics)

    @tf.function
    def backward(start_time, end_time, final_state, final_loss_gradient):
        final_phase_point = [final_state, final_loss_gradient]
        for var in variables:
            zeros = tf.zeros_like(var)
            final_phase_point.append(zeros)
        ode_final_value = forward(start_time=end_time,
                                  end_time=start_time,
                                  initial_phase_point=final_phase_point)
        init_state, init_loss_gradient, *grad_loss_by_vars = ode_final_value
        return init_state, init_loss_gradient, grad_loss_by_vars

    return backward
