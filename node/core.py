"""Core algorithm implementations and utils."""

import tensorflow as tf
from node.utils.nest import nest_map


class Backward:

  def __call__(self, start_time, end_time, final_state, final_loss_gradient):
    """
    Parameters
    ----------
    start_time : Time
    end_time : Time
    final_state : PhasePoint
      The $z^{\alpha}(t_N)$ in the paper. The final outputs of the
      Neural ODE.
    final_loss_gradient : PhasePoint
      The $\frac{\partial L}{\partial z^{\alpha}(t_N)}$ in the paper.
      The gradient of loss by the final output of the Neural ODE
      (i.e. by the `final_state`).

    Returns
    -------
    (initial_state, gradient_by_state, gradient_by_variables)
      initial_state : PhasePoint
      gradient_by_state : PhasePoint
      gradient_by_variables : list of tensor
        In the paper, they're $z(t_0)$, $\partial L / \partial z^{\alpha}(t_0)$,
        and $\partial L / \partial \theta_i^{\alpha}$, respectively.
    """
    return NotImplemented


def reverse_mode_derivative(ode_solver, network, variables):
  """Implements the algorithm 1 in the paper original paper (1806.07366).

  Parameters
  ----------
  ode_solver : ODESolver
  network : PhaseVectorField
    The $f(x, t)$ in the paper.
  variables: list of tf.Variable
    The $\theta$ in the paper. In practice, it's a list of variables.
    Thus $\theta = (\theta_1, \ldots)$,

  Returns
  -------
  Backward
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
    ode_result = forward(end_time,
                         start_time,
                         final_phase_point)
    init_state, init_loss_gradient, *grad_loss_by_vars = ode_result.phase_point
    return init_state, init_loss_gradient, list(grad_loss_by_vars)

  return backward


@nest_map
def _negate(x):
  return -1 * x


def get_dtype_from_signature(signature):
  """
  Parameters
  ----------
  signature : nest structure of tf.TensorSpec

  Returns
  -------
  tf.Dtype
  """
  spec = signature[0]
  while not isinstance(spec, tf.TensorSpec):
    spec = spec[0]
  dtype = spec.dtype
  return dtype


def get_node_function(solver, fn, signature=None):
  r"""

  ```math

  Let $f$ a phase vector field, then defnine the "node function" $F$ as

  $$ F(t0, t1, x0) := x0 + \int_{t_0}^{t_1} f(t, F(t0, t, x0)) dt. $$

  That is, the ending phase point at $t1$ of the flow starting on $x0$ at $t_0$
  on the phase vector field.

  ```

  Parameters
  ----------
  solver : ODESolver
  fn : PhaseVectorField
    The $f$ in the definition.
  signature : nest structure of tf.TensorSpec
    The signature of the phase point `x` in the definition, optional

  Returns
  -------
  PhaseVectorField
  """
  forward = solver(fn)

  if signature:
    dtype = get_dtype_from_signature(signature)
    t0_spec = tf.TensorSpec(shape=[], dtype=dtype)
    t1_spec = tf.TensorSpec(shape=[], dtype=dtype)
    input_signature = [t0_spec, t1_spec] + signature
  else:
    input_signature = None

  @tf.function(input_signature=input_signature)
  def node_fn(t0, t1, x0):
    """
    Parameters
    ----------
    t0: Time
    t1: Time
    x0: PhasePoint

    Returns
    -------
    PhasePoint
    """

    @tf.custom_gradient
    def custom_gradient_fn(*x):
      r"""For matching the signature of `tf.custom_gradient`
      https://tensorflow.google.cn/api_docs/python/tf/custom_gradient

      Explicitly, the inputs to this function shall be tensors, each as
      one arg; the output `grad_fn` accepts tensors, each as one arg, and
      kwargs involving the key "variables".

      To make this API compatible with phase point, which is a nested structure
      of tensors, we have to flatten the phase point before passing into this
      function, and nest back within this function.
      """
      # nest back the flatten phase point to the original
      x = tf.nest.pack_sequence_as(x0, list(x))

      ode_result = forward(t0, t1, x)
      y = ode_result.phase_point

      # TF will catch all the variables watched by `tf.GradientTape`,
      # and pass them into the `grad_fn` via the `variables` kwarg.
      @tf.function
      def grad_fn(*grad_ys, **kwargs):
        grad_ys = tf.nest.pack_sequence_as(y, list(grad_ys))

        # XXX: `tf.custom_gradient` has an unfixed
        # [bug](https://github.com/tensorflow/tensorflow/issues/31945).
        # Because of this, temporally, we need some [hacking]
        # (https://github.com/tensorflow/tensorflow/issues/31945#issuecomment-545801180)  # noqa:E501
        # TODO: Re-write this part when the bug is fixed.
        variables = kwargs.get('variables', None)

        backward = reverse_mode_derivative(solver, fn, variables)
        _, grad_by_x, grad_by_vars = backward(t0, t1, y, grad_ys)
        return [grad_by_x], grad_by_vars

      return y, grad_fn

    return custom_gradient_fn(*tf.nest.flatten(x0))

  return node_fn


def get_dynamical_node_function(
    dynamical_solver, static_solver, fn, stop_condition, signature=None):
  """Generalization of the `get_node_function` for dynamical solver.

  Parameters
  ----------
  dynamical_solver : DynamicalODESolver
  static_solver : ODESolver
  fn : PhaseVectorField
    The $f$ in the definition.
  signature : nest structure of tf.TensorSpec, optional
    The signature of the phase point `x` in the definition.

  Returns
  -------
  PhaseVectorField
  """
  forward = dynamical_solver(fn, stop_condition)

  if signature:
    dtype = get_dtype_from_signature(signature)
    t0_spec = tf.TensorSpec(shape=[], dtype=dtype)
    input_signature = [t0_spec] + signature
  else:
    input_signature = None

  @tf.function(input_signature=input_signature)
  def node_fn(t0, x0):
    """
    Parameters
    ----------
    t0 : Time
    x0 : PhasePoint

    Returns
    -------
    PhasePoint
    """

    @tf.custom_gradient
    def custom_gradient_fn(*x):
      r"""For matching the signature of `tf.custom_gradient`
      https://tensorflow.google.cn/api_docs/python/tf/custom_gradient

      Explicitly, the inputs to this function shall be tensors, each as
      one arg; the output `grad_fn` accepts tensors, each as one arg, and
      kwargs involving the key "variables".

      To make this API compatible with phase point, which is a nested structure
      of tensors, we have to flatten the phase point before passing into this
      function, and nest back within this function.
      """
      # nest back the flatten phase point to the original
      x = tf.nest.pack_sequence_as(x0, list(x))

      ode_result = forward(t0, x)
      t1 = ode_result.time
      y = ode_result.phase_point

      # TF will catch all the variables watched by `tf.GradientTape`,
      # and pass them into the `grad_fn` via the `variables` kwarg.
      @tf.function
      def grad_fn(*grad_ys, **kwargs):
        grad_ys = tf.nest.pack_sequence_as(y, list(grad_ys))

        # XXX: `tf.custom_gradient` has an unfixed
        # [bug](https://github.com/tensorflow/tensorflow/issues/31945).
        # Because of this, temporally, we need some [hacking]
        # (https://github.com/tensorflow/tensorflow/issues/31945#issuecomment-545801180)  # noqa:E501
        # TODO: Re-write this part when the bug is fixed.
        variables = kwargs.get('variables', None)

        backward = reverse_mode_derivative(solver, fn, variables)
        _, grad_by_x, grad_by_vars = backward(t0, t1, y, grad_ys)
        return [grad_by_x], grad_by_vars

      return y, grad_fn

    return custom_gradient_fn(*tf.nest.flatten(x0))

  return node_fn
