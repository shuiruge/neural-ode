"""Core algorithm implementations and utils."""

import tensorflow as tf


def reverse_mode_derivative(ode_solver, network, variables):
    r"""Implements the algorithm 1 in the paper original paper (1806.07366).

    Args:
        ode_solver: ODESolver
        network: PhaseVectorField
            The $f(x, t)$ in the paper.
        variables: List[tf.Variable]
            The $\theta$ in the paper. In practice, it's a list of variables.
            Thus $\theta = (\theta_1, \ldots)$,

    Returns: Callable for backward propagation
        Args:
            start_time: Time
            end_time: Time
            final_state: PhasePoint
                The $z^{\alpha}(t_N)$ in the paper. The final outputs of the
                Neural ODE.
            final_loss_gradient: PhasePoint
                The $\frac{\partial L}{\partial z^{\alpha}(t_N)}$ in the paper.
                The gradient of loss by the final output of the Neural ODE
                (i.e. by the `final_state`).
        Returns: Tuple[PhasePoint, PhasePoint, List[tf.Tensor]]
            For the initial state, the gradient of loss by the initial state,
            and the gradient of loss by the variables `variables`. In the
            paper, they are $z(t_0)$, $\partial L / \partial z^{\alpha}(t_0)$,
            and $\partial L / \partial \theta_i^{\alpha}$, respectively.
    """

    @tf.function
    def aug_dynamics(time, aug_phase_point):
        state, adjoint, *_ = aug_phase_point
        neg_adjoint = nest_map(lambda x: -1 * x, adjoint)

        with tf.GradientTape() as g:
            g.watch(state)
            output = network(time, state)
        # According to
        # # https://www.tensorflow.org/api_docs/python/tf/custom_gradient
        # `tf.gradients` or `g.gradient`, if the third argument is filled,
        # returns the vector-Jacobian-products directly. In fact, TF
        # implements VJP inside, and compute gradients via VJP.
        vjps = g.gradient(output, [state] + variables, neg_adjoint)

        new_aug_phase_point = [output] + vjps
        return new_aug_phase_point

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


def get_node_function(solver, t0, fn):
    r"""Converts a phase vector field `f(t, x)` to `F(t, x)` which is defined
    as $ F(t, x) = \int_{t_0}^t f(t, F(t, x)) dt $, where $ F(t_0, x) = x $.
    That is, the ending phase point at `t` of the flow starting on `x` at `t0`
    on the phase vector field.

    Args:
        solver: ODESolver
        t0: Time
            The start time of the phase flow.
        fn: PhaseVectorField

    Returns: PhaseVectorField
    """
    forward = solver(fn)

    @tf.function
    def node_fn(t, x):
        """
        Args:
            t: Time
            x: PhasePoint

        Returns: PhasePoint
        """

        # XXX: `tf.custom_gradient` has its own bug, which affects the code
        # here. C.f. https://github.com/tensorflow/tensorflow/issues/31945
        # Because of this, currently, `node` does NOT support multi-inputs.
        @tf.custom_gradient
        def custom_gradient_fn(x):
            """For matching the signature of `tf.custom_gradient`
            https://tensorflow.google.cn/api_docs/python/tf/custom_gradient
            """
            y = forward(t0, t, x)

            # TF will catch all the variables watched by `tf.GradientTape`,
            # and pass them into the `grad_fn` via the `variables` kwarg.
            @tf.function
            def grad_fn(grad_ys, variables=None):
                backward = reverse_mode_derivative(solver, fn, variables)
                _, grad_by_x, grad_by_vars = backward(t0, t, y, grad_ys)
                return [grad_by_x], grad_by_vars

            return y, grad_fn

        return custom_gradient_fn(x)

    return node_fn


def nest_map(fn, *args):  # TODO: add example.
    """All args shall share the same nesting structure."""
    _check_same_structure(*args)

    if not _is_nested(args[0]):
        return fn(*args)

    if isinstance(args[0], tuple):
        return tuple(nest_map(fn, *subargs) for subargs in zip(*args))
    elif isinstance(args[0], list):
        return list(nest_map(fn, *subargs) for subargs in zip(*args))
    else:
        raise ValueError('XXX')


def _is_nested(x):
    """Auxillary function of `nest_map`."""
    return isinstance(x, (list, tuple))


def _check_same_structure(*args):
    first_arg, *rest_args = args
    for arg in rest_args:
        tf.nest.assert_same_structure(first_arg, arg)


def tracer(solver, fn):
    """
    Args:
        solver: ODESolver
        fn: PhaseVectorField

    Returns: Callable[[Time, Time, Time, tf.Tensor], tf.TensorArray]
        The arguments are start time, end time, time difference, and
        initial phase point. Returns the trajectory.
    """
    forward = solver(fn)

    @tf.function
    def trace(t0, t1, dt, x):
        dt = tf.where(t1 > t0, dt, -dt)
        num_grids = int((t1 - t0) / dt + 1)
        ts = tf.linspace(t0, t1, num_grids)

        i = 0
        xs = tf.TensorArray(x.dtype, size=num_grids)
        xs = xs.write(i, x)

        ts = tf.linspace(t0, t1, num_grids)
        for t in ts[:-1]:
            x = forward(t, t + dt, x)
            i += 1
            xs = xs.write(i, x)
        return xs.stack()

    return trace
