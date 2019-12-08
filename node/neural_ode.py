import tensorflow as tf

def reverse_mode_derivative(network, var_list, interval,
                            final_state, loss_gradient):
    
    def aug_dynamics(time, phase_point):
        state, adjoint, var_list = phase_point
        with tf.GradientTape(persistent=True) as t:
            output = network(time, state)
        grad_state = t.gradients(output, state)
        grad_vars = t.gradients(output, var_list)
        del t
        new_state = [
            output,
            -tf.inner(adjoint, grad_state),
            [-tf.inner(adjoint, _) for _ in grad_vars]]
        return new_state

    return NotImplemented
