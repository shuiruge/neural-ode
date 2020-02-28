import tensorflow as tf


class Inspector(tf.keras.callbacks.Callback):
  r"""Inspects the model in the training process.

  Args:
    skip_step: int
      Logging the weights values per `skip_step` steps (batches).
  """

  def __init__(self, skip_step=10, **kwargs):
    super().__init__(**kwargs)
    self.skip_step = skip_step

    self.logs = []  # type: [dict]

  def inspect(self):
    """Function that produces the values to be inspected. These values shall be
    explicit like `np.array`, instead of abstract objects like `tf.Tensor`.

    Returns: dict
    """
    return NotImplemented

  def on_train_batch_end(self, batch, logs=None):
    if batch % self.skip_step == 0:
      self.logs.append(self.inspect())


class WeightInspector(Inspector):
  """Inspects the weights values while training."""

  def inspect(self):
    vars = self.model.trainable_variables
    return [self._get_value(var) for var in vars]

  def _get_value(self, tensor):
    return {'name': tensor.name,
            'value': tensor.numpy().tolist()}


class LayerInspector(Inspector):
  r"""Inspects the activations and gradients of all layers in the
  training process.

  Suppose `self.model` has layers: layer_1, layer_2, ..., layer_n,
  then activations are the activations of each layer: z_1, z_2, ...
  z_n. And the gradients are dL / dz_1, dL / d_2, ..., dL / dz_n.

  Args:
    samples: Tuple[np.array, np.array]
      The samples used for passing through the layers of the model.
      The same type as the inputs in `model.fit`.
    level: str
      "original": logs all components of the tensors
      "mean_and_std": logs only means and stds of the tensors
    skip_step: int
      Logging the weights values per `skip_step` steps (batches).
  """

  def __init__(self, samples, level='original', skip_step=10, **kwargs):
    super().__init__(skip_step, **kwargs)
    self.samples = samples
    self.level = level

    self._X = tf.convert_to_tensor(samples[0])
    self._y = tf.convert_to_tensor(samples[1])

  def inspect(self):
    layers = self.model.layers

    loss = self.model.loss
    if isinstance(self.model.loss, str):
      loss = tf.keras.losses.get(loss)

    with tf.GradientTape(persistent=True) as g:
      activations = []
      activation = self._X
      for layer in layers:
        activation = layer(activation)
        activations.append(activation)

      output = activations[-1]
      loss_val = loss(self._y, output)

    grad = g.gradient(
        loss_val,
        output,
        unconnected_gradients='zero')
    gradients = [grad]

    num_layers = len(layers)
    for i in range(num_layers - 1):
      grad = g.gradient(
          activations[(num_layers - i - 1)],
          activations[(num_layers - i - 2)],
          grad,
          unconnected_gradients='zero')
      gradients.append(grad)

    activations = [self._get_value(t) for t in activations]
    gradients = [self._get_value(t) for t in gradients]
    gradients.reverse()

    return {'activations': activations,
            'gradients': gradients}

  def _get_value(self, tensor):
    array = tensor.numpy()

    if self.level == 'original':
      return {'value': array.tolist()}

    elif self.level == 'mean_and_std':
      return {'mean': array.mean(), 'std': array.std()}

    else:
      raise ValueError(f'Unknown level: "{self.level}".')
