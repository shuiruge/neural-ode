from typing import List
import numpy as np
import tensorflow as tf


class Inspector(tf.keras.callbacks.Callback):
  """
  Args:
    inspect_fn: Callable[[], List[np.array]]
      Function that produces the values to be inspected. These values shall be
      explicit like `np.array`, instead of abstract objects like `tf.Tensor`.
    skip_step: int
      Logging the weights values per `skip_step` steps (batches).
  """

  def __init__(self, inspect_fn, skip_step, **kwargs):
    super().__init__(**kwargs)
    self.inspect_fn = inspect_fn
    self.skip_step = skip_step

    self.logs = None  # type: List[List[np.array]]

  def on_train_batch_end(self, batch, logs=None):
    if batch % self.skip_step == 0:
      self._add_to_logs(self.inspect_fn())

  def _add_to_logs(self, values):
    if self.logs is None:
      self.logs = [[] for _ in values]

    for log, value in zip(self.logs, values):
      log.append(value)


class WeightInspector(Inspector):
  """Logging the weights values while training.

  Args:
    skip_step: int
      Logging the weights values per `skip_step` steps (batches).
  """

  def __init__(self, skip_step, **kwargs):

    def inspect_fn():
      vars = self.model.trainable_variables
      return [var.numpy() for var in vars]

    super().__init__(inspect_fn, skip_step, **kwargs)
