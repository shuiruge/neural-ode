import tensorflow as tf


class WeightsLogger(tf.keras.callbacks.Callback):
  """Logging the weights values while training.

  Args:
    skip_step: int
      Logging the weights values per `skip_step` steps (batches).
  """

  def __init__(self, skip_step, **kwargs):
    super().__init__(**kwargs)
    self.skip_step = skip_step

    self.weights = None

  def on_train_batch_end(self, batch, logs=None):
    vars = self.model.trainable_variables
    if self.weights is None:
      self.weights = [[] for _ in vars]

    if batch % self.skip_step == 0:
      for logs, var in zip(self.weights, vars):
        logs.append(var.numpy())
