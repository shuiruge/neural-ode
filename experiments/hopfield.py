"""Utils for experimenting on the continuum of Hopfield."""

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from node.core import get_node_function, get_dynamical_node_function
from node.solvers.runge_kutta import RK4Solver, RKF56Solver
from node.solvers.dynamical_runge_kutta import (
  DynamicalRK4Solver, DynamicalRKF56Solver)
from node.hopfield import hopfield, get_stop_condition


# for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


SOLVER = 'rk4'
# SOLVER = 'rkf56'


class NodeLayer(tf.keras.layers.Layer):

  def __init__(self, fn, t, dt, **kwargs):
    super().__init__(**kwargs)

    self._config = {'fn': fn, 't': t, 'dt': dt}

    self.t = tf.convert_to_tensor(t)
    self.dt = tf.convert_to_tensor(dt)

    if SOLVER == 'rk4':
      solver = RK4Solver(self.dt)
    elif SOLVER == 'rkf56':
      solver = RKF56Solver(self.dt, tol=1e-2, min_dt=1e-1)
    else:
      raise ValueError()

    t0 = tf.constant(0.)
    self._node_fn = get_node_function(solver, t0, lambda _, x: fn(x))

  def call(self, x):
    y = self._node_fn(self.t, x)
    return y

  def get_config(self):
    config = super().get_config()
    for k, v in self._config.items():
      config[k] = v
    return config


class HopfieldLayer(tf.keras.layers.Layer):

  def __init__(self, fn, t0, dt, lower_bounded_fn, **kwargs):
    super().__init__(**kwargs)

    self._config = {'fn': fn, 't0': t0, 'dt': dt,
                    'lower_bounded_fn': lower_bounded_fn}

    self.fn = fn
    self.lower_bounded_fn = lower_bounded_fn

    self.t0 = tf.convert_to_tensor(t0)
    self.dt = tf.convert_to_tensor(dt)

    if SOLVER == 'rk4':
      solver = RK4Solver(self.dt)
      dynamical_solver = DynamicalRK4Solver(self.dt)
    elif SOLVER == 'rkf56':
      solver = RKF56Solver(self.dt, tol=1e-2, min_dt=1e-1)
      dynamical_solver = DynamicalRKF56Solver(self.dt, tol=1e-2, min_dt=1e-1)
    else:
      raise ValueError()

    t0 = tf.constant(0.)
    energy = lambda x: lower_bounded_fn(fn(x))
    pvf = hopfield(energy)
    stop_condition = get_stop_condition(pvf, max_delta_t=5, tolerance=1e-1)

    self.energy = energy
    self.pvf = pvf
    self._node_fn = get_dynamical_node_function(
      dynamical_solver, solver, pvf, stop_condition)

  def call(self, x0):
    y = self._node_fn(self.t0, x0)
    return y

  def get_config(self):
    config = super().get_config()
    for k, v in self._config.items():
      config[k] = v
    return config


def process(X, y):
  X = X / 255.
  X = np.where(X > 0.5, 1., 0.)
  X = np.reshape(X, [-1, 28 * 28])
  y = np.eye(10)[y]
  return X.astype('float32'), y.astype('float32')


class HopfieldModel(tf.keras.Sequential):

  def __init__(self, lower_bounded_fn, units, t, dt, layerized=False, **kwargs):
    self._config = {'lower_bounded_fn': lower_bounded_fn, 'units': units,
                    't': t, 'dt': dt, 'layerized': layerized}

    self.lower_bounded_fn = lower_bounded_fn
    self.units = units
    self.t = t
    self.dt = dt
    self.layerized = layerized

    # XXX:
    # notice that the output activation of `fn` shall not be ReLU
    # and the like, which are always non-negative. This will cause
    # numerical instability when doing integral (for ODE). Indeed,
    # using ReLU leads to NaN in practice.
    fn = tf.keras.Sequential([
      # tf.keras.layers.Dense(256, activation='relu'),  # test!
      tf.keras.layers.Dense(1024, activation='relu'),
      tf.keras.layers.Dense(units, activation=None)])

    layers = [
      tf.keras.Input([28 * 28]),
      tf.keras.layers.Dense(units, use_bias=False),  # down-sampling
    ]

    if layerized:
      num_node_layers = int(t / dt)
      for _ in range(num_node_layers):
        layers.append(NodeLayer(fn, dt, dt))
      layers.append(HopfieldLayer(fn, dt, dt, lower_bounded_fn))
    else:
      layers.append(HopfieldLayer(fn, t, dt, lower_bounded_fn))

    layers.append(tf.keras.layers.Dense(10, activation='softmax'))

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


def get_longer_period_model(model, t):
  # build longer period model
  longer_period_model = HopfieldModel(
    model.lower_bounded_fn, model.units, t, model.dt, layerized=False)
  longer_period_model.compile(
    loss='categorical_crossentropy',
    metrics=['accuracy'])

  # use the trained weights
  longer_period_model.set_weights(model.get_weights())

  return longer_period_model


def longer_period_effect(model, t, x_train, y_train):
  longer_period_model = get_longer_period_model(model, t)

  base_score = model.evaluate(x_train, y_train, verbose=0)
  longer_period_score = longer_period_model.evaluate(
    x_train, y_train, verbose=0)

  print('Longer period effect:')
  for i in range(2):
    print(f'{base_score[i]:.5f} => {longer_period_score[i]:.5f}')


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


def white_noise(binary, scale):
  noised = binary + scale * np.random.normal(size=binary.shape)
  noised = np.where(noised > 1, np.ones_like(noised), noised)
  noised = np.where(noised < 0, np.zeros_like(noised), noised)
  return noised


def add_white_noise(scalar, scale):

  def add_noise(x):
    # get noised_x_train
    x_orig = scalar.inverse_transform(x)
    noised_x_orig = white_noise(x_orig, scale)
    noised_x = scalar.transform(noised_x_orig)
    return noised_x

  return add_noise


def noised_effect(model, add_noise, x_train, y_train):
  base_score = model.evaluate(x_train, y_train, verbose=0)

  noised_x_train = add_noise(x_train)
  noised_score = model.evaluate(noised_x_train, y_train, verbose=0)

  print('Noised effect:')
  for i in range(2):
    print(f'{base_score[i]:.5f} => {noised_score[i]:.5f}')


def get_non_node_model(node_model, x_train, y_train):
  layers = []
  for layer in node_model.layers:
    if isinstance(layer, HopfieldLayer):
      units = layer.output_shape[-1]
      layers.append(tf.keras.layers.Dense(1024, activation='relu'))
      layers.append(tf.keras.layers.Dense(units, activation='relu'))
      continue

    if isinstance(layer, tf.keras.layers.Dense):
      config = layer.get_config()
      units = config['units']
      activation = config['activation']
      layers.append(tf.keras.layers.Dense(units, activation=activation))

  non_node_model = tf.keras.Sequential(layers)
  non_node_model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.optimizers.Nadam(1e-3),
    metrics=['accuracy'])
  non_node_model.fit(x_train, y_train,
                     epochs=10, batch_size=128)
  return non_node_model


if __name__ == '__main__':

  mnist = tf.keras.datasets.mnist
  (x_train, y_train), _ = mnist.load_data()
  x_train, y_train = process(x_train, y_train)

  scalar = StandardScaler()
  scalar.fit(x_train)
  x_train = scalar.transform(x_train)

  @tf.function
  def lower_bounded_fn(x):
    return 5 * tf.sqrt(tf.reduce_sum(tf.square(x), axis=1))

  model = HopfieldModel(
    lower_bounded_fn, units=64, t=1.0, dt=0.1, layerized=False)
  model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.optimizers.Adam(1e-3, epsilon=1e-2),
    metrics=['accuracy'])

  model.fit(x_train, y_train,
            epochs=10, batch_size=128)

  # longer_period_effect(model, 3.0, x_train, y_train)

  # # test noise effect

  # add_noise = add_white_noise(scalar, scale=0.03)
  # noised_effect(model, add_noise, x_train, y_train)

  # longer_period_model = get_longer_period_model(model, t=3.0)
  # noised_effect(longer_period_model, add_noise, x_train, y_train)

  # non_node_model = get_non_node_model(model, x_train, y_train)
  # noised_effect(non_node_model, add_noise, x_train, y_train)
