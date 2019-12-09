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

    def __call__(self,
                 phase_vector_field,
                 start_time,
                 end_time,
                 initial_phase_point):
        """
        Args:
            phase_vector_field: PhaseVectorField
            start_time: Time
            end_time: Time
            initial_phase_point: PhasePoint

        Returns: PhasePoint
        """
        return NotImplemented


def reverse_mode_derivative(ode_solver,
                            network,
                            var_list,
                            start_time,
                            end_time,
                            final_state,
                            loss_gradient,
                            batch_axes):
    r"""Implements the algorithm 1 in the paper original paper (1806.07366).

    Args:
        ode_solver: ODESolver
        network: Callable[[tf.Tensor], tf.Tensor]
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
        batch_axis: Collection[int]
            The collection of axis of batch for the tensor passing through
            the Neural ODE.

    Returns: tf.Tensor
        The $\frac{\partial L}{\partial \theta_i}$ for the variable
        $\theta_i$ in the variable list $\theta$.
    """
    inner_prod_axis = [i for i in final_state.shape
                       if i not in batch_axes]

    def inner_prod(tensor, other_tensor):
        return tf.reduce_sum(tensor * other_tensor, inner_prod_axis)

    def aug_dynamics(time, phase_point):
        state, adjoint, *var_list = phase_point

        with tf.GradientTape() as t:
            t.watch(state)
            output = network(time, state)
        grad_state, *grad_vars = t.gradient(output, [state] + var_list)

        new_state = output
        new_adjoint = -inner_prod(adjoint, grad_state)
        new_var_list = [-inner_prod(adjoint, grad_var)
                        for grad_var in grad_vars]
        new_phase_point = [new_state, new_adjoint] + new_var_list
        return new_phase_point

    ode_initial_value = ([final_state, loss_gradient] +
                         [tf.zeros_like(var) for var in var_list])
    ode_final_value = ode_solver(
        aug_dynamics, start_time, end_time, ode_initial_value)
    init_state, init_loss_gradient, *grad_loss_by_vars = ode_final_value
    return grad_loss_by_vars
