import tensorflow as tf


class GlorotUniform(tf.keras.initializers.GlorotUniform):
    """Glorot uniform initializer with scale kwarg."""

    def __init__(self, scale=None, seed=None):
        super().__init__(seed)
        self.scale = 1 if scale is None else scale

    def __call__(self, *args, **kwargs):
        return self.scale * super().__call__(*args, **kwargs)
