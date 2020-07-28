import tensorflow as tf


@tf.function
def binarize(x, num_bits):
  """
  Parameters
  ----------
  x : tensor
    In range [-1, +1].
  num_bits: int

  Returns
  -------
  tensor
    Shape: `x.shape + [num_bits]`, values in {-1, +1}.
  """

  def binarize_recur(x, bits):
    if len(bits) == num_bits:
      return tf.stack(bits, axis=-1)
    return binarize_recur(
      tf.where(x > 0, x * 2 - 1, x * 2 + 1),
      bits + [tf.where(x > 0, 1., -1.)])

  return binarize_recur(x, [])


@tf.function
def inverse_binarize(x):
  """Inverse function of `binarize`.

  Parameters
  ----------
  x : tensor
    In range {-1, +1}.

  Returns
  -------
  tensor
    Shape: `x.shape[:-1]`, values in [-1, +1].
  """
  num_bits = x.get_shape().as_list()[-1]
  bits = tf.constant([0.5 ** (i + 1) for i in range(num_bits)])
  bits = tf.expand_dims(bits, axis=-1)  # shape: (num_bits, 1)
  return tf.squeeze(tf.matmul(x, bits), axis=-1)
