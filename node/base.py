import tensorflow as tf
from typing import Union, List, Callable

Time = float
PhasePoint = Union[tf.Tensor, List[tf.Tensor]]
PhaseVectorField = Callable[[PhasePoint, Time], PhasePoint]


class ODESolver:
    r"""
    Definition
    ----------
    $\text{ode_solver}(f, t_0, t_1, z(t_0)) := z(t_0) + \int_{t_0}^{t_1} f(z(t), t) dt$  # noqa:E501
    which is exectly the $z(t_1)$.
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


# TODO: re-write
def reverse_mode_derivative(ode_solver,
                            network,
                            var_list,
                            start_time,
                            end_time,
                            final_state,
                            loss_gradient):
    r"""Implements the algorithm 1 in the paper original paper (1806.07366).

    Args:
        ode_solver: ODESolver
        network: Callable[[tf.Tensor, Time], tf.Tensor]
            The $f(x, t)$ in the paper.
        var_list: List[tf.Variable]
            The $\theta$ in the paper. In practice, it's a list of variables.
            Thus $\theta = (\theta_1, \ldots)$,
        start_time: Time
        end_time: Time
        final_state: tf.Tensor
            The $z^{\alpha}(t_1)$ in the paper. The final outputs of the
            Neural ODE.
        loss_gradient: tf.Tensor
            The $\frac{\partial L}{\partial z^{\alpha}(t_1)}$ in the paper.
            The gradient of loss by the final output of the Neural ODE
            (i.e. by the `final_state`).

    Returns: tf.Tensor
        The $\frac{\partial L}{\partial \theta_i}$ for the variable
        $\theta_i$ in the variable list $\theta$.
    """
    # collect the axis that are not for batch dimension
    non_batch_axis = [i for i, _ in enumerate(final_state.shape)][1:]

    def contract(adjoint, jacob, is_var):
        r"""$a_{\alpha} \frac{\partial f^{\alpha}}{\partial b^{\beta}}$
        where $b$ can either be $h$ or $\theta_i$."""  # XXX: wrong.
        extra_rank = len(jacob.shape) - len(adjoint.shape)
        adjoint = tf.reshape(adjoint, (adjoint.shape.as_list() +
                                       [1] * extra_rank))
        if is_var:
            axis = [0] + non_batch_axis
        else:
            axis = non_batch_axis
        return tf.reduce_sum(adjoint * jacob, axis)

    def aug_dynamics(phase_point, time):
        state, adjoint, *_ = phase_point  # adjoint: [B, D]

        with tf.GradientTape() as g:
            g.watch(state)
            output = network(state, time)
        # According to
        # # https://www.tensorflow.org/api_docs/python/tf/custom_gradient
        # `tf.gradients` or `g.gradient`, if the third argument is filled,
        # returns the vector-Jacobian-products directly. In fact, TF
        # implements VJP inside, and compute gradients via VJP.
        grads = g.gradient(output, [state] + var_list,
                           output_gradients=adjoint)

        new_phase_point = [output] + grads
        return new_phase_point

    final_phase_point = [final_state, loss_gradient]
    for var in var_list:
        zeros = tf.zeros_like(var)
        final_phase_point.append(zeros)
    ode_final_value = ode_solver(
        aug_dynamics, end_time, start_time, final_phase_point)
    init_state, init_loss_gradient, *grad_loss_by_vars = ode_final_value
    return grad_loss_by_vars
