import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from typing import Dict, Any


class Hook:

  def __call__(self, inputs: tf.Tensor, outputs: tf.Tensor):
    return NotImplemented


class LayerWithHooks(Layer):
  """
  C.f. https://github.com/tensorflow/tensorflow/issues/33478#issuecomment-568290488
  """
  def __init__(self, layer: Layer, hooks: [Hook] = None, **kwargs):
    super().__init__(**kwargs)
    self._layer = layer
    self._hooks = hooks or []

  def call(self, inputs):
    outputs = self._layer(inputs)
    for hook in self._hooks:
      hook(inputs, outputs)
    return outputs

  def register_hook(self, hook: Hook):
    self._hooks.append(hook)

  @property
  def hooks(self):
    return self._hooks


class InputOutputSaver(Hook):

  def __init__(self):
    self.inputs: tf.Tensor = NotImplemented
    self.outputs: tf.Tensor = NotImplemented

  def __call__(self, inputs, outputs):
    self.inputs = inputs
    self.outputs = outputs


class TracedLayer(LayerWithHooks):

  def __init__(self, layer, **kwargs):
    hooks = [InputOutputSaver()]
    if kwargs.get('name', None) is None:
      kwargs = kwargs.copy()
      kwargs['name'] = f'traced_{layer.name}'
    super().__init__(layer, hooks, **kwargs)

  @property
  def trace(self):
    saver = self.hooks[0]
    return (saver.inputs, saver.outputs)


class TracedModel(tf.keras.Model):

  def __init__(self, model: tf.keras.Model, **kwargs):
    super().__init__(**kwargs, name=f'traced_{model.name}')
    self._model = model

    self._traced_layers = []
    for layer in model.layers:
      self._traced_layers.append(TracedLayer(layer))

  def call(self, inputs):
    x = inputs
    for layer in self._traced_layers:
      x = layer(x)
    return x

  def build(self, batch_input_shape):
    self._model.build(batch_input_shape)

  @property
  def trace(self):
    return [layer.trace for layer in self.traced_layers]

  @property
  def traced_layers(self):
    return self._traced_layers


def get_loss(model: tf.keras.Model) -> tf.losses.Loss:
  if isinstance(model.loss, tf.losses.Loss):
    loss = model.loss

  elif isinstance(model.loss, str):
    # may be sample-wize loss function
    loss_fn = tf.keras.losses.get(model.loss)

    def loss(y_true, y_pred):
      return tf.reduce_mean(loss_fn(y_true, y_pred))

  else:
    raise ValueError(f'{model.loss}')

  return loss


def get_layerwise_gradients(model, inputs, targets):
  assert isinstance(model, TracedModel)

  with tf.GradientTape(persistent=True) as g:
    predictions = model(inputs)
    for inputs, outputs in model.trace:
      g.watch(inputs)
      g.watch(outputs)
    loss = get_loss(model)(targets, predictions)

  gradients = []
  for x, y in model.trace:
    gradients.append(
      (g.gradient(loss, x), g.gradient(loss, y)))
  return gradients


def get_optimizer_variables(model):
  return model.optimizer.variables


def get_layer_activations(model):
  assert isinstance(model, TracedModel)
  layers = model.traced_layers
  activations = [outputs for inputs, outputs in model.trace]
  return {layer.name: act for layer, act in zip(layers, activations)}


def get_weights(model):
  return model.trainable_variables


def get_weight_gradiants(model, inputs, targets):
  with tf.GradientTape() as g:
    loss = get_loss(model)(targets, predictions)
  return g.gradients(loss, get_weights(model))


class Inspector(tf.keras.callbacks.Callback):
  """Inspects the activations and gradients acrossing layers."""

  def __init__(self,
               inspect_activations: bool,
               inspect_gradients: bool,
               inspect_weights: bool,
               inspect_weight_gradients: bool):
    self.inspect_activations = inspect_activations
    self.inspect_gradients = inspect_gradients
    self.inspect_weights = inspect_weights
    self.inspect_weight_gradients = inspect_weight_gradients

    self._inspection_report: Dict[str, tf.Tensor] = {}

  def get_aspects(self, tensor: tf.Tensor) -> Dict[str, np.array]:
    return NotImplemented

  def _inspect(self):
    if self.inspect_activations: 
      self._inspection_report['activations'] = get_activations(self.model)
    if self.inspect_gradients:
      self._inspection_report['gradients'] = get_layerwise_gradients
