import tensorflow as tf


class GlorotUniform(tf.keras.initializers.GlorotUniform):
  """Glorot uniform initializer with scale kwarg."""

  def __init__(self, scale=None, **kwargs):
    super().__init__(**kwargs)
    scale = 1 if scale is None else scale
    self.scale = tf.convert_to_tensor(scale)

  def __call__(self, *args, **kwargs):
    return self.scale * super().__call__(*args, **kwargs)
