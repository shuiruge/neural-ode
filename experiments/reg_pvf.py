import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from node.core import get_node_function
from node.solvers.runge_kutta import RK4Solver, RKF56Solver


# for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


tf.keras.backend.clear_session()


# SOLVER = 'rk4'
SOLVER = 'rkf56'
DEBUG = 1


class NodeLayer(tf.keras.layers.Layer):

  def __init__(self, t0, t1, dt, units, hidden_units, regular_factor,
               name='NodeLayer', **kwargs):
    super().__init__(name=name, **kwargs)
    self.t0 = tf.convert_to_tensor(t0)
    self.t1 = tf.convert_to_tensor(t1)
    self.dt = tf.convert_to_tensor(dt)
    self.units = units
    self.hidden_units = hidden_units
    self.regular_factor = regular_factor

    if SOLVER == 'rk4':
      self._solver = RK4Solver(self.dt)
    elif SOLVER == 'rkf56':
      self._solver = RKF56Solver(self.dt, tol=1e-2, min_dt=1e-1)
    else:
      raise ValueError()

    self._fn = tf.keras.Sequential([
      tf.keras.Input([units]),
      tf.keras.layers.Dense(hidden_units, activation='relu'),
      tf.keras.layers.Dense(units)])

    self._node_fn = tf.function(
      get_node_function(self._solver, lambda t, x: self._fn(x)))

  def call(self, x0):
    t0 = self.t0
    t1 = self.t1
    dt = (self.t1 - self.t0) / 5.
    x = x0
    xs = []
    for i in range(5):
      x = self._node_fn(t0, t1, x)
      xs.append(x)
      t0 = t1
      t1 = t1 + dt
    pvfs = [self._fn(x) for x in xs]
    # add regularization for ensuring stability
    regularization = 1 / 5. * self.regular_factor * sum(
       tf.reduce_mean(tf.abs(pvf)) for pvf in pvfs)
    self.add_loss(regularization, inputs=True)
    return xs[0]


def process_data(X, y):
  X = X / 255.
  y = np.eye(10)[y]
  return X.astype('float32'), y.astype('float32')


def random_flip(binary, flip_ratio):

  def flip(binary):
    return np.where(binary > 0.5,
                    np.zeros_like(binary),
                    np.ones_like(binary))

  if_flip = np.random.random(size=binary.shape) < flip_ratio
  return np.where(if_flip, flip(binary), binary)


def add_random_flip_noise(flip_ratio):

  def add_noise(x):
    return random_flip(x, flip_ratio)

  return add_noise


def add_gaussian_noise(scale):

  def add_noise(x):
    y = x + scale * np.random.normal(size=x.shape)
    return np.clip(y, 0, 1)

  return add_noise


def add_pixalwise_gaussian_noise(factor, scale):

  def add_noise(x):
    noise = scale * np.random.normal(size=x.shape)
    noise = np.where(np.random.random(size=x.shape) < factor,
                     noise,
                     np.zeros_like(x))
    y = x + noise
    return np.clip(y, 0, 1)

  return add_noise


def get_benchmark_model(dataset, units, hidden_units):
  model = tf.keras.Sequential([
    tf.keras.layers.Reshape([28 * 28]),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(units, activation='relu'),
    tf.keras.layers.Dense(hidden_units, activation='relu'),
    tf.keras.layers.Dense(units),
    tf.keras.layers.Dense(10, activation='softmax'),
  ])
  model.compile(
    loss=tf.losses.CategoricalCrossentropy(),
    optimizer=tfa.optimizers.Lookahead(tf.optimizers.Adam(1e-3)),
    metrics=['accuracy'],
  )
  model.fit(dataset, verbose=2, epochs=10)
  return model


def get_model(dataset, units, hidden_units, t, regular_factor):
  model = tf.keras.Sequential([
    tf.keras.layers.Reshape([28 * 28]),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(units),
    NodeLayer(0., t, 0.1, units, hidden_units, regular_factor),
    tf.keras.layers.Dense(10, activation='softmax')
  ])
  model.compile(
    loss=tf.losses.CategoricalCrossentropy(),
    optimizer=tfa.optimizers.Lookahead(tf.optimizers.Adam(1e-3)),
    metrics=['accuracy'],
  )
  model.fit(dataset, verbose=2, epochs=10)
  return model


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = process_data(x_train, y_train)
x_test, y_test = process_data(x_test, y_test)

dataset = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
           .batch(128))


units = 128
hidden_units = 512
benchmark_model = get_benchmark_model(dataset, units, hidden_units)

t = 5.
regular_factor = 1.
model = get_model(dataset, units, hidden_units, t, regular_factor)

noised_x_train = add_pixalwise_gaussian_noise(0.3, 0.3)(x_train)
y_pred_benchmark = benchmark_model.predict(noised_x_train)
y_pred = model.predict(noised_x_train)

def get_acc(y_true, y_pred):
  return tf.reduce_mean(
    tf.metrics.categorical_accuracy(y_true, y_pred))

print(get_acc(y_train, y_pred_benchmark))
print(get_acc(y_train, y_pred))
