import tensorflow as tf
from node.utils.layer_norm import layer_norm


def get_albert_dynamics(self_attention, feed_forward):
  r"""
  ```math

  Let $f_{SA}(.)$ the self-attention (without mask) and $f_{FF}(.)$ the
  feed-forward. Then the ALBERT update is

  \begin{align}

    z_{t+1}^a & = \
      \text{layernorm}^a \left( x_t + \Delta t f_{SA} (x_t, m_t) \right) \\

    x_{t+1}^a & = \
      \text{layernorm}^a \left( z_t + \Delta t f_{FF}(z_t) \right).

    m_{t+1}^a & = m_{t}

  \end{align}

  If $\Delta t \rightarrow 0$, and re-denote $x_t \rightarrow x_1(t)$,
  $z_t \rightarrow x_2(t)$, $m_t \rightarrow x_3(t)$, then we gain the
  dynamics

  \begin{align}

    \frac{dx_1^a}{dt} & = \
      \nabla_b \text{layernrom}^a (x_3) f_{FF}^b (x_3); \\

    \frac{dx_2^a}{dt} & = 0 \\

    \frac{dx_3^a}{dt} & = \
      \nabla_b \text{layernrom}^a (x_1) f_{SA}^b (x_1, x_2).

  \end{align}

  ```

  Args:
    self_attention: Callable[[tf.Tensor, tf.Tensor], tf.Tensor]
    feed_forward: Callable[[tf.Tensor], tf.Tensor]
      The first input tensor of `self_attention` and the input tensor of
      `feed_forward` share the same shape: `[..., sequence_length, depth]`.
      The second input tensor of `self_attention` has the shape
      `[..., sequence_length, 1]`. All tensors share the same dtype.

  Returns: PhaseVectorField
    The phase space consists three tensors, the two inputs of `self_attention`
    and the input of `feed_forward`.
  """

  @tf.function
  def albert_dynamics(t, x):
    x1, x2, x3 = x

    def layer_norm_vjp(x, y):
      # \sum_b \frac{ \partial \text{layernrom}^a }{ \partial x^b } y^b
      # (notice the symmetry of the gradient of layer-norm)
      with tf.GradientTape() as g:
        g.watch(x)
        ln = layer_norm(x)
      return g.gradient(ln, x, y, unconnected_gradients='zero')

    dx1 = layer_norm_vjp(x3, feed_forward(x3))
    dx2 = tf.zeros_like(x2)
    dx3 = layer_norm_vjp(x1, self_attention(x1, x2))
    return [dx1, dx2, dx3]

  return albert_dynamics
