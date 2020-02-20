from typing import List
import numpy as np
import tensorflow as tf


class Inspector(tf.keras.callbacks.Callback):
  r"""Inspects the model in the training process.

  Args:
    inspect_fn: Callable[[], List[np.array]]
      Function that produces the values to be inspected. These values shall be
      explicit like `np.array`, instead of abstract objects like `tf.Tensor`.
    skip_step: int
      Logging the weights values per `skip_step` steps (batches).
  """

  def __init__(self, inspect_fn, skip_step=10, **kwargs):
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
  """Inspects the weights values while training."""

  def __init__(self, **kwargs):

    def inspect_fn():
      vars = self.model.trainable_variables
      return [var.numpy() for var in vars]

    super().__init__(inspect_fn, **kwargs)


class ActivationInspector(Inspector):
  r"""Inspects the activations of all layers in the training process.

  Args:
    samples:
      The samples used for passing through the layers of the model.
      The same type as the inputs in `model.fit`.
  """

  def __init__(self, samples, **kwargs):

    def inspect_fn():
      activations = []
      activation = samples
      for layer in self.model.layers:
        activation = layer(activation)
        activations.append(activation.numpy())
      return activations

    super().__init__(inspect_fn, **kwargs)
