r"""
## CONCLUSION

  * It seems that it is the bad data point that leads to the instability.

  * It seems that, in addition, it is the optimizer that leads to the
    instability. In fact, using Adamax instead of using Adam ceases this
    problem. Using a `tfa.optimizers.Lookahead` wrapper for Adam (or Adamax)
    is a better choice.

"""

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from node.core import get_dynamical_node_function
from node.solvers.runge_kutta import RK4Solver, RKF56Solver
from node.solvers.dynamical_runge_kutta import (
  DynamicalRK4Solver, DynamicalRKF56Solver)
from node.utils.rmsprop import rmsprop
from node.utils.nest import nest_map
from node.hopfield import hopfield, get_stop_condition
from node.utils.layers import (TracedModel, get_layer_activations,
                               get_layerwise_gradients, get_weights,
                               get_weight_gradiants, get_optimizer_variables)


# for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


tf.keras.backend.clear_session()


SOLVER = 'rk4'
# SOLVER = 'rkf56'


class HopfieldLayer(tf.keras.layers.Layer):

  def __init__(self, fn, dt, lower_bounded_fn, name='HopfieldLayer',
               **kwargs):
    super().__init__(name=name, **kwargs)

    self._config = {'fn': fn, 'dt': dt,
                    'lower_bounded_fn': lower_bounded_fn}

    self.fn = fn
    self.lower_bounded_fn = lower_bounded_fn

    self.dt = tf.convert_to_tensor(dt)

    if SOLVER == 'rk4':
      solver = RK4Solver(self.dt)
      dynamical_solver = DynamicalRK4Solver(self.dt)
    elif SOLVER == 'rkf56':
      solver = RKF56Solver(self.dt, tol=1e-2, min_dt=1e-1)
      dynamical_solver = DynamicalRKF56Solver(self.dt, tol=1e-2, min_dt=1e-1)
    else:
      raise ValueError()

    def energy(x):
      return lower_bounded_fn(fn(x))

    pvf = hopfield(energy)

    max_delta_t=100.0
    tolerance=1e-1

    def stop_condition(t0, x0, t, x):
      if tf.abs(t - t0) > max_delta_t:
        tf.print('hahaha')  # XXX: test!
        return True
      max_abs_velocity = tf.reduce_max(tf.abs(pvf(t, x[0])))
      if max_abs_velocity < tolerance:
        tf.print('------------!!!!!!!!!!!!!!!!!!---------------')  # XXX: test!
        return True
      tf.print('max_vel - ', max_abs_velocity)  # XXX: test!
      return False

    def pvf_2(t, x):
      x = rmsprop(pvf, eps=1e-3)(t, x)
      return x

    self.energy = energy
    self.pvf = pvf_2
    self._node_fn = get_dynamical_node_function(
      dynamical_solver, solver, pvf_2, stop_condition)

  def call(self, x0):
    t0 = tf.constant(0.)

    ms = nest_map(lambda x: tf.ones_like(x))(x0)
    x0 = (x0, ms)

    y = self._node_fn(t0, x0)

    y = y[0]

    max_abs = tf.reduce_max(tf.math.abs(y))
    threshold = tf.constant(10.)
    regularizer = tf.math.maximum(max_abs, threshold) - threshold
    self.add_loss(regularizer, inputs=True)

    return y

  def get_config(self):
    config = super().get_config()
    for k, v in self._config.items():
      config[k] = v
    return config


class HopfieldModel(tf.keras.Sequential):

  def __init__(self, lower_bounded_fn, units, dt, hidden_units=1024,
               **kwargs):
    self._config = {'lower_bounded_fn': lower_bounded_fn, 'units': units,
                    'dt': dt, 'hidden_units': hidden_units}

    self.lower_bounded_fn = lower_bounded_fn
    self.units = units
    self.dt = dt
    self.hidden_units = hidden_units

    # XXX:
    # notice that the output activation of `fn` shall not be ReLU
    # and the like, which are always non-negative. This will cause
    # numerical instability when doing integral (for ODE). Indeed,
    # using ReLU leads to NaN in practice.
    fn = tf.keras.Sequential([
        tf.keras.Input([units]),
        tf.keras.layers.Dense(hidden_units, activation='relu'),
        tf.keras.layers.Dense(units, activation=None)
      ], name='HopfieldFn')

    layers = [
      tf.keras.Input([28 * 28]),
      tf.keras.layers.Dense(units, name='DownSampling'),
      tf.keras.layers.LayerNormalization(name='DownSamplingLayerNorm'),
    ]

    layers.append(HopfieldLayer(fn, dt, lower_bounded_fn))

    layers.append(tf.keras.layers.Dense(10, name='OutputLogits'))

    super().__init__(layers=layers, **kwargs)

    # `tf.keras.Model` will trace the variables within the layers'
    # `trainable_variables` attributes. The variables initialized
    # outside the layers will not be traced, e.g. those of the `fn`.
    #
    # Extra variables are appended into the property `trainable_variables`
    # of `tf.keras.Model` via the `_trainable_weights` attribute.
    self._trainable_weights += fn.trainable_variables

  def get_config(self):
    config = super().get_config()
    for k, v in self._config.items():
      config[k] = v
    return config


def process_data(X, y):
  X = X / 255.
  X = np.where(X > 0.5, 1., 0.)
  X = np.reshape(X, [-1, 28 * 28])
  y = np.eye(10)[y]
  return X.astype('float32'), y.astype('float32')


def random_flip(binary, flip_ratio):

  def flip(binary):
    return np.where(binary > 0.5,
                    np.zeros_like(binary),
                    np.ones_like(binary))

  if_flip = np.random.random(size=binary.shape) < flip_ratio
  return np.where(if_flip, flip(binary), binary)


def add_random_flip_noise(scalar, flip_ratio):

  def add_noise(x):
    # get noised_x_train
    x_orig = scalar.inverse_transform(x)
    noised_x_orig = random_flip(x_orig, flip_ratio)
    noised_x = scalar.transform(noised_x_orig)
    return noised_x

  return add_noise


def add_white_noise(noise_scale):

  def add_noise(x):
    return x + np.random.normal(scale=noise_scale, size=x.shape)

  return add_noise


def noised_effect(model, add_noise, x_train, y_train):
  base_score = model.evaluate(x_train, y_train, verbose=0)

  noised_x_train = add_noise(x_train)
  noised_score = model.evaluate(noised_x_train, y_train, verbose=0)

  print('Noised effect:')
  for i in range(2):
    print(f'{base_score[i]:.5f} => {noised_score[i]:.5f}')


def remove_outliers(array, min_val, max_val):
  ids = []
  high_quality_samples = []
  for i, sample in enumerate(array):
    if sample.min() >= min_val and sample.max() < max_val:
      ids.append(i)
      high_quality_samples.append(sample)
  return np.array(ids), np.stack(high_quality_samples)


def inspect(model, inputs, targets):
  inspection_report = {}

  inspection_report['inputs'] = {
    'max': inputs.numpy().max(),
    'min': inputs.numpy().min(),
  }

  inspection_report['loss'] = loss.numpy()

  dws = get_weight_gradiants(model, X, y_true)
  grads = get_layerwise_gradients(model, X, y_true)
  activations = get_layer_activations(model)
  opt_vars = model.optimizer.variables()

  def describe_tensor(tensor):
    return {
      'maxabs': np.abs(tensor.numpy()).max(),
      'minabs': np.abs(tensor.numpy()).min(),
      'mean': tensor.numpy().mean(),
      'std': tensor.numpy().std(),
    }

  inspection_report['weight gradients'] = {
    w.name: describe_tensor(dw)
    for w, dw in zip(model.trainable_variables, dws)
  }

  inspection_report['layerwise gradients'] = {
    layer.name: {
      'grad_by_inputs': describe_tensor(grad_x),
      'grad_by_outputs': describe_tensor(grad_y),
    }
    for layer, (grad_x, grad_y) in zip(model.traced_layers, grads)
  }

  inspection_report['layerwise activations'] = {
    layer.name: describe_tensor(activation)
    for layer, activation in zip(model.traced_layers, activations)
  }

  inspection_report['optimizer variables'] = {
    w.name: describe_tensor(w)
    for w in opt_vars
  }
  return inspection_report


# ---------------------------- doing experiments below -------------------------


mnist = tf.keras.datasets.mnist
(x_train, y_train), _ = mnist.load_data()
x_train, y_train = process_data(x_train, y_train)

scalar = StandardScaler()
scalar.fit(x_train)
x_train = scalar.transform(x_train)
print(x_train.shape, y_train.shape)

# ids, x_train = remove_outliers(x_train, -3, 3)
# y_train = y_train[ids]

x_train, y_train = shuffle(x_train, y_train, random_state=SEED)

# XXX: test!
num_data = 1000
x_train = x_train[:num_data]
y_train = y_train[:num_data]

print(x_train.shape, y_train.shape)

@tf.custom_gradient
def sqrt(x):
  """Numerical stable sqrt function."""
  y = tf.sqrt(x)

  def grad_fn(dy):
    eps = 1e-8
    return 0.5 * dy / (y + eps)

  return y, grad_fn

@tf.function
def lower_bounded_fn(x):
  return 5 * sqrt(tf.reduce_sum(tf.square(x), axis=1))

model = HopfieldModel(
  lower_bounded_fn, units=64, dt=1e-2, hidden_units=1024)
model = TracedModel(model)
model.compile(
  loss=tf.losses.CategoricalCrossentropy(from_logits=True),
  # optimizer=tf.optimizers.Adamax(1e-3),
  optimizer=tfa.optimizers.Lookahead(tf.optimizers.Adamax(1e-3)),
  metrics=['accuracy'],
)

# use custom training loop for the convenience of doing experiments
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

previous_loss = None
inspection_reports = []

epochs = 20
for epoch in range(epochs):
  print(f'EPOCH {epoch + 1}/{epochs}')

  for step, (X, y_true) in enumerate(dataset.batch(128)):
    print(step)
    with tf.GradientTape() as g:
      y_pred = model(X)
      loss = model.loss(y_true, y_pred)
    grads = g.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

    if previous_loss is None or loss < 2 * previous_loss:
      previous_loss = loss
    else:
      # raise ValueError()
      pass

  # model.evaluate(x_train, y_train, verbose=2)

  # if step % 5 == 0:
  #   inspection_reports.append(inspect(model, X, y_true))

noised_effect(model, add_white_noise(0.5), x_train, y_train)
