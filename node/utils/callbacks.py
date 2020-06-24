import tensorflow as tf


class InspectResult:

  def __init__(self, batch, loss, activations, gradients, weights,
               weight_gradients, extra_info=None):
    self.batch = int(batch)
    self.loss = float(loss)
    self.activations = activations
    self.gradients = gradients
    self.weights = weights
    self.weight_gradients = weight_gradients
    self.extra_info = extra_info


class LayerInspector(tf.keras.callbacks.Callback):
  r"""Inspects the activations and gradients acrossing layers in the
  training process.

  The model shall be `tf.keras.Sequential`.

  WARNINGS:
  This callback will slow down the training process. It will cause
  `WARNING:tensorflow:Method (on_train_batch_end) is slow compared `
  `to the batch update`.

  Args:
    samples: Tuple[np.array, np.array]
      The samples used for passing through the layers of the model.
      The same type as the inputs in `model.fit`.
    aspects: Callable[[np.array], dict]
      Defines what aspects of the tensor (casted to `np.array`) are
      you to inspect.
    skip_step: int
      Logging the weights values per `skip_step` steps (batches).
    **kwargs:
      Kwargs of `tf.keras.callbacks.Callback`.
  """

  def __init__(self, samples, aspects, skip_step, **kwargs):
    super().__init__(**kwargs)
    self.samples = samples
    self.aspects = aspects
    self.skip_step = skip_step

    self._X = tf.convert_to_tensor(samples[0])
    self._y = tf.convert_to_tensor(samples[1])

    self.logs = []

  def on_train_batch_end(self, batch, logs=None):
    if batch % self.skip_step == 0:
      log = self._inspect(batch)
      self.logs.append(log)

  def on_train_begin(self, logs=None):
    assert isinstance(self.model, tf.keras.Sequential)

  def _inspect(self, batch):
    layers = self.model.layers

    # get loss
    if isinstance(self.model.loss, str):
      loss = tf.keras.losses.get(self.model.loss)
    else:
      loss = self.model.loss

    variables = self.model.trainable_variables

    activations = []
    gradients = []

    # forward propagate
    x = self._X
    gradient_triplets = []
    for layer in layers:
      with tf.GradientTape() as g:
        g.watch(x)
        y = layer(x)
      activations.append({layer.name: self.aspects(y.numpy())})
      gradient_triplets.append((g, x, y))
      x = y

    # compute dloss / dy
    with tf.GradientTape() as g:
      g.watch(y)
      l = loss(self._y, y)
    grad = g.gradient(l, y, unconnected_gradients='zero')

    name = f'{layers[-1].name} -> Loss'
    gradients.append({name: self.aspects(grad.numpy())})

    # backward propagate
    i = len(layers) - 1
    for g, x, y in gradient_triplets[::-1]:
      grad = g.gradient(y, x, grad, unconnected_gradients='zero')

      if i - 1 < 0:
        name = f'Input -> {layers[0].name}'
      else:
        name = f'{layers[i - 1].name} -> {layers[i].name}'
      gradients.append({name: self.aspects(grad.numpy())})
      i -= 1

    gradients = gradients[::-1]

    # to compute the dL / dv for v in variables, we have to do
    # forward and backward computations once again
    variables = self.model.trainable_variables
    with tf.GradientTape() as g:
      l = loss(self._y, self.model(self._X))
    grad_vars = g.gradient(l, variables, unconnected_gradients='zero')
    weight_gradients = {v.name: self.aspects(g.numpy())
                        for v, g in zip(variables, grad_vars)}

    loss_val = l.numpy()

    weights = {v.name: self.aspects(v.numpy()) for v in variables}

    return InspectResult(batch, loss_val, activations, gradients, weights,
                         weight_gradients)
