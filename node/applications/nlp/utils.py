import tensorflow as tf


def swapaxes(x, axis1, axis2):
  """
  Examples:
  >>> x = ...  # shape: [..., 2, 3, 4]
  >>> swapaxes(x, -3, -2)  # shape: [..., 3, 2, 4]

  Args:
    x: tf.Tensor
    axis1: int
    axis2: int

  Returns: tf.Tensor
  """
  rank = len(x.get_shape().as_list())
  if axis1 < 0:
    axis1 = rank + axis1
  if axis2 < 0:
    axis2 = rank + axis2

  perm = []
  for axis in range(rank):
    if axis == axis1:
      perm.append(axis2)
    elif axis == axis2:
      perm.append(axis1)
    else:
      perm.append(axis)

  return tf.transpose(x, perm)


def reshape_last_axes(x, shape, num_axes):
  """
  Examples:
  >>> x = ...  # shape: [..., m, n]
  >>> reshape_last_axes(x, [m * n], 2)  # shape [..., (m * n)]
  >>> x = ...  # shape: [..., (m * n)]
  >>> reshape_last_axes(x, [m, n], 1)  # shape [..., m, n]

  Args:
    x: tf.Tensor
    shape: Iterable[int]
    num_axes: int

  Returns: tf.Tensor
  """
  orig_shape = tf.shape(x)
  new_shape = tf.concat(
      [orig_shape[:(-num_axes)], shape], axis=0)
  return tf.reshape(x, new_shape)


def attention(q, k, v, mask=None):
  """Calculates the attention weights. Luong-style attention.

  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead)
  but it must be broadcastable for addition.

  Args:
    q: tf.Tensor
      Shape: [..., seq_len_q, depth]
    k: tf.Tensor
      Shape: [..., seq_len_k, depth]
    v: tf.Tensor
      Shape: [..., seq_len_v, depth_v]
    mask: Optional[tf.Tensor]
      Float tensor with shape broadcastable to [..., seq_len_q, seq_len_k].
      Defaults to None.

  Returns: Tuple[tf.Tensor, tf.Tensor]
    output: shape [..., seq_len_q, depth]
    attention_weights: shape [..., seq_len_q, seq_len_k]
  """
  # (..., seq_len_q, seq_len_k)
  matmul_qk = tf.matmul(q, k, transpose_b=True)

  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], matmul_qk.dtype)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  # (..., seq_len_q, seq_len_k)
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
  """
  Args:
    d_model: int
    num_heads: int
    attention: Callable
      Args:
        q: tf.Tensor
          Shape: [..., seq_len_q, depth].
        k: tf.Tensor
          Shape: [..., seq_len_k, depth].
        v: tf.Tensor
          Shape: [..., seq_len_v, depth_v], seq_len_k = seq_len_v.
        mask: Optional[tf.Tensor]
          Float tensor with shape broadcastable to [..., seq_len_q, seq_len_k].
          Defaults to None.
      Returns: Tuple[tf.Tensor, tf.Tensor]
        output: shape [..., seq_len_q, depth];
        attention_weights: shape [..., seq_len_q, seq_len_k].
  """

  def __init__(self, d_model, num_heads, attention=attention):
    super().__init__()
    self.num_heads = num_heads
    self.d_model = d_model
    self.attention = attention

    assert d_model % num_heads == 0

    self.depth = d_model // num_heads

    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

    self.dense = tf.keras.layers.Dense(d_model)

  def _split_heads(self, x):
    """Split the last dimension into (num_heads, depth).

    Args:
      x: tf.Tensor
        Shape [..., seq_len, d_model].

    Returns: tf.Tensor
      Shape [..., num_heads, seq_len, depth].
    """
    # [..., seq_len, num_heads, depth]
    x = reshape_last_axes(x, [self.num_heads, self.depth], 1)
    # [..., num_heads, seq_len, depth]
    x = swapaxes(x, -3, -2)
    return x

  def _concat_heads(self, x):
    """Inverse of `self._split_heads`.

    Args:
      x: tf.Tensor
        Shape: [..., num_heads, seq_len, depth].

    Returns: tf.Tensor
      Shape: [..., seq_len, (num_heads * depth)]
    """
    # [..., seq_len, num_heads, depth]
    x = swapaxes(x, -3, -2)
    # [..., seq_len, (num_heads * depth)]
    x = reshape_last_axes(x, [self.num_heads * self.depth], 2)
    return x

  def call(self, inputs):
    """
    Args:
      inputs:
        v: tf.Tensor
          Shape: [..., seq_len_v, v_input_dims], where the `v_input_dims`
          can be arbitrary.
        k: tf.Tensor
          Shape: [..., seq_len_k, k_input_dims], where the `k_input_dims`
          can be arbitrary. The `seq_len_k == seq_len_v`.
        q: tf.Tensor
          Shape: [..., seq_len_q, q_input_dims], where the `q_input_dims`
          can be arbitrary.
        mask: tf.Tensor
          Float tensor with shape broadcastable to
          [..., num_heads, seq_len_q, seq_len_k].

    Returns: tf.Tensor
      Shape: [batch_size, seq_len_q, d_model].
    """
    v, k, q, mask = inputs
    q = self.wq(q)  # [..., seq_len, d_model]
    k = self.wk(k)  # [..., seq_len, d_model]
    v = self.wv(v)  # [..., seq_len, d_model]

    q = self._split_heads(q)  # [..., num_heads, seq_len_q, depth]
    k = self._split_heads(k)  # [..., num_heads, seq_len_k, depth]
    v = self._split_heads(v)  # [..., num_heads, seq_len_v, depth]

    # [..., num_heads, seq_len_q, depth]
    scaled_attention, _ = self.attention(q, k, v, mask)

    # [batch_size, seq_len_q, d_model]
    concat_attention = self._concat_heads(scaled_attention)
    # [..., seq_len_q, d_model]
    output = self.dense(concat_attention)
    return output


class SelfAttention(MultiHeadAttention):

  def call(self, inputs):
    """
    Args:
      inputs:
        x: tf.Tensor
          Shape: [..., seq_len, x_dims].
        mask: tf.Tensor
          Float tensor with shape broadcastable to
          [..., num_heads, seq_len, seq_len].

    Returns: tf.Tensor
      Shape: [..., seq_len, d_model].
    """
    x, mask = inputs
    return super().call([x, x, x, mask])


class PositionalEncoding(tf.keras.layers.Layer):
  """
  Experiments:
    For making the (embedding) input and the positional encoding the same
    order, the input is layer-normalized, initially to unit std, which is
    the same order as the positional encoding. The kernel and bias parameters
    in the layer-normalization help determine the weight of the positional
    information by training.

  Args:
    d_model: int
      Dimension of the model.
    max_position: int
      Any number that is greater than the max sequence length of the inputs
      of the model.
  """

  def __init__(self, d_model, max_position):
    super().__init__()
    self.max_position = max_position

    self._pos_encoding = positional_encoding(max_position, d_model)
    self._factor = tf.math.sqrt(tf.cast(d_model, self.dtype))
    self._layer_norm = tf.keras.layers.LayerNormalization()

  def call(self, x):
    """
    Args:
      x: tf.Tensor
        Shape: [..., sequence_length, x_dims].

    Returns: tf.Tensor
      The same shape and dtype as the `x`.
    """
    seq_len = tf.shape(x)[1]
    # if seq_len > self.max_position:
    #   raise ValueError('The sequence length of input shall be no greater '
    #                    f'than max position {self.max_position}, '
    #                    f'but found {seq_len}.')
    pos_enc = self._pos_encoding[:seq_len, :]

    broadcast_shape = tf.concat(
        [tf.ones([tf.rank(x) - 2], 'int32'), tf.shape(x)[-2:]],
        axis=0)
    pos_enc = tf.reshape(pos_enc, broadcast_shape)
    # return x * self._factor + pos_enc
    # return x + pos_enc  # XXX: test!
    return self._layer_norm(x) + pos_enc  # XXX: test!


def get_angles(pos, i, d_model):
  """Auxillary function of `PositionalEncoding`.

  Args:
    position: int
    i: int
    d_model: int

  Returns: float
  """
  pos = tf.cast(pos, 'float32')
  i = tf.cast(i, 'int32')
  d_model = tf.cast(d_model, 'float32')
  p = tf.cast(2 * (i // 2), 'float32') / d_model
  angle_rates = 1. / tf.math.pow(10000., p)
  return pos * angle_rates


def positional_encoding(position, d_model):
  """Auxillary function of `PositionalEncoding`.

  Args:
    position: int
    d_model: int

  Returns: tf.Tensor
    Shape [position, d_model]
  """
  # [position, d_model]
  angles = get_angles(
      tf.range(position)[:, tf.newaxis],
      tf.range(d_model)[tf.newaxis, :],
      d_model)

  # apply sin to even indices in the array; 2i
  sin = tf.sin(angles[:, 0::2])
  # apply cos to odd indices in the array; 2i+1
  cos = tf.cos(angles[:, 1::2])

  # [position, d_model]
  pos_encoding = tf.reshape(tf.stack([sin, cos], axis=-1),
                            [position, d_model])
  return pos_encoding
