"""Core algorithm implementations and utils."""

import tensorflow as tf
from node.utils.nest import nest_map


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
    neg_adjoint = _negate(adjoint)

    with tf.GradientTape() as g:
      g.watch(state)
      output = network(time, state)
    # According to
    # # https://www.tensorflow.org/api_docs/python/tf/custom_gradient
    # `tf.gradients` or `g.gradient`, if the third argument is filled,
    # returns the vector-Jacobian-products directly. In fact, TF
    # implements VJP inside, and compute gradients via VJP.
    vjps = g.gradient(output, [state] + variables, neg_adjoint,
                      unconnected_gradients='zero')

    new_aug_phase_point = [output] + vjps
    return new_aug_phase_point

  forward = ode_solver(aug_dynamics)

  @tf.function
  def backward(start_time, end_time, final_state, final_loss_gradient):
    final_phase_point = [final_state, final_loss_gradient]
    for var in variables:
      zeros = tf.zeros_like(var)
      final_phase_point.append(zeros)
    ode_final_value = forward(end_time,
                              start_time,
                              final_phase_point)
    init_state, init_loss_gradient, *grad_loss_by_vars = ode_final_value
    return init_state, init_loss_gradient, list(grad_loss_by_vars)

  return backward


@nest_map
def _negate(x):
  return -1 * x


def get_node_function(solver, t0, fn, signature=None):
  r"""

  ```math

  Let $f$ a phase vector field, then defnine the "node function" $F$ as

  $$ F(t, x) := \int_{t_0}^t f(t, F(t, x)) dt, $$ and
  
  $$ F(t_0, x) = x. $$
  
  That is, the ending phase point at $t$ of the flow starting on $x$ at $t_0$
  on the phase vector field.

  ```

  Args:
    solver: ODESolver
    t0: Time
      The start time of the phase flow. The $t_0$ in the definition.
    fn: PhaseVectorField
      The $f$ in the definition.
    signature: Nest[tf.TensorSpec]
      The signature of the phase point `x` in the definition.

  Returns: PhaseVectorField
  """
  forward = solver(fn)

  def process_args(args):
    if len(args) == 1:  # single arg in `**args`
      return args[0]
    else:
      # `args` is a tuple, we need list instead for unifying
      # the nesting structures
      return list(args)

  if signature is None:
    input_signature = None
  else:
    time_signature = tf.TensorSpec(shape=[], dtype=t0.dtype)
    input_signature = [time_signature] + signature

  @tf.function(input_signature=input_signature)
  def node_fn(t, x):
    """
    Args:
      t: Time
      x: PhasePoint

    Returns: PhasePoint
    """
    @tf.custom_gradient
    def custom_gradient_fn(*x):
      """For matching the signature of `tf.custom_gradient`
      https://tensorflow.google.cn/api_docs/python/tf/custom_gradient

      Explicitly, the inputs to this function shall be tensors, each as
      one arg; the output `grad_fn` accepts tensors, each as one arg, and
      kwargs involving the key "variables".
      """
      x = process_args(x)

      y = forward(t0, t, x)

      # TF will catch all the variables watched by `tf.GradientTape`,
      # and pass them into the `grad_fn` via the `variables` kwarg.
      @tf.function
      def grad_fn(*grad_ys, **kwargs):
        # XXX: `tf.custom_gradient` has an unfixed
        # [bug](https://github.com/tensorflow/tensorflow/issues/31945).
        # Because of this, temporally, we need some [hacking]
        # (https://github.com/tensorflow/tensorflow/issues/31945#issuecomment-545801180)  # noqa:E501
        # TODO: Re-write this part when the bug is fixed.
        grad_ys = process_args(grad_ys)
        variables = kwargs.get('variables', None)

        backward = reverse_mode_derivative(solver, fn, variables)
        _, grad_by_x, grad_by_vars = backward(t0, t, y, grad_ys)
        return [grad_by_x], grad_by_vars

      return y, grad_fn

    if isinstance(x, list):
      return custom_gradient_fn(*x)
    else:
      return custom_gradient_fn(x)

  return node_fn
