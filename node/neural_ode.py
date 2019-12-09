import tensorflow as tf


def reverse_mode_derivative(odeint, network, var_list, interval,
                            final_state, loss_gradient, batch_axes):
    r"""Implements the algorithm 1 in the paper original paper (1806.07366).

    Args:
        odeint: Callable[[PhaseVectorField, List[Time], PhasePoint],
                         PhasePoint]
        network: Callable[[tf.Tensor], tf.Tensor]
        var_list: List[tf.Variable]
            The $\theta$ in the paper. In practice, it's a list of variables.
            Thus $\theta = (\theta_1, \ldots)$,
        interval: List[float]
            Works for fix grid ODE solvers.
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
    ode_final_value = odeint(aug_dynamics, interval, ode_initial_value)
    init_state, init_loss_gradient, *grad_loss_by_vars = ode_final_value
    return grad_loss_by_vars
