import tensorflow as tf


def reverse_mode_derivative(odeint, network, var_list, interval,
                            final_state, loss_gradient, batch_axes):
    """Implements the algorithm 1 in the paper original paper (1806.07366).

    Args:
        odeint: Callable[[PhaseVectorField, List[Time], PhasePoint],
                         PhasePoint]
        network: Callable[[tf.Tensor], tf.Tensor]
        var_list: List[tf.Variable]
        interval: List[float]
        final_state: tf.Tensor
            The $z^{\alpha}(t_1)$ in the paper.
        loss_gradient: tf.Tensor
            The $\frac{\partial L}{\partial z^{\alpha}(t_1)}$ in the paper.
        batch_axis: Collection[int]
    
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
        return [new_state, new_adjoint] + new_var_list

    ode_initial_value = ([final_state, loss_gradient] +
                         [tf.zeros_like(var) for var in var_list])
    ode_final_value = odeint(aug_dynamics, interval, ode_initial_value)
    init_state, init_loss_gradient, *grad_loss_by_vars = ode_final_value
    return grad_loss_by_vars
