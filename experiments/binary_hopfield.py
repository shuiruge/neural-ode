import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from node.core import get_dynamical_node_function
from node.solvers.runge_kutta import RK4Solver, RKF56Solver
from node.solvers.dynamical_runge_kutta import (
  DynamicalRK4Solver, DynamicalRKF56Solver)


# for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


tf.keras.backend.clear_session()


# SOLVER = 'rk4'
SOLVER = 'rkf56'
DEBUG = 1


@tf.custom_gradient
def inverse_boundary_reflect(x):
  r"""Inverse of $G$ function in the reference [1].

  $$ G^{-1}: \mathbb{R}^n \mapsto [0, 1]^n $$

  References:
    1. Diffusions for Global Optimization, S. Geman and C. Hwang
  """
  i = tf.cast(x, 'int32')
  delta = x - tf.cast(i, x.dtype)
  y = tf.where(i % 2 == 0, x, 1 - x)

  def grad_fn(dy):
    return tf.where(i % 2 == 0, dy, -dy)

  return y, grad_fn


class StopCondition(tf.Module):

  def __init__(self, energy, max_delta_t, rel_tol):
    super().__init__()
    self.energy = energy
    self.max_delta_t = max_delta_t
    self.rel_tol = rel_tol

    self._previous_energy = tf.Variable(
      0., shape=tf.TensorShape(None), validate_shape=False,
      trainable=False)

  @tf.function
  def __call__(self, t0, x0, t, x):
    shall_stop = False
    current_energy = self.energy(x)
    if tf.abs(t - t0) > self.max_delta_t:
      shall_stop = True
      if DEBUG > 0:
        tf.print('reached max delta t:', self.max_delta_t)
    elif t > t0:
      delta_energy = current_energy - self._previous_energy
      rel_delta = (tf.math.abs(delta_energy) /
                   (tf.math.abs(self._previous_energy) + 1e-8))
      shall_stop = tf.reduce_max(rel_delta) < self.rel_tol
      if shall_stop and DEBUG > 0:
        tf.print('relaxed at delta t:', t - t0)
    else:
      pass
    if DEBUG > 1:
      x_mean, x_var = tf.nn.moments(x, axes=[1])
      if DEBUG > 2:
        tf.print('mean:', x_mean)
      tf.print('std:', tf.sqrt(x_var))
    self._previous_energy.assign(current_energy)
    return shall_stop


class HopfieldLayer(tf.keras.layers.Layer):

  def __init__(self, energy, dt, stop_condition,
               name='HopfieldLayer', **kwargs):
    super().__init__(name=name, **kwargs)

    self.energy = energy
    self.dt = tf.convert_to_tensor(dt)
    self.stop_condition = stop_condition

    if SOLVER == 'rk4':
      solver = RK4Solver(self.dt)
      dynamical_solver = DynamicalRK4Solver(self.dt)
    elif SOLVER == 'rkf56':
      solver = RKF56Solver(self.dt, tol=1e-2, min_dt=1e-1)
      dynamical_solver = DynamicalRKF56Solver(self.dt, tol=1e-2, min_dt=1e-1)
    else:
      raise ValueError()

    def pvf(_, x):
      with tf.GradientTape() as g:
        g.watch(x)
        bounded_x = inverse_boundary_reflect(x)
        e = energy(bounded_x)
      return -g.gradient(e, x, unconnected_gradients='zero')

    self.pvf = pvf
    self._node_fn = get_dynamical_node_function(
      dynamical_solver, solver, pvf, stop_condition)

  @tf.function
  def call(self, x0):
    t0 = tf.constant(0.)
    x1 = self._node_fn(t0, x0)
    return x1


class AveragePool2D(tf.keras.layers.Layer):

  def __init__(self, strides, **kwargs):
    super().__init__(**kwargs)
    self._pooling = tf.keras.layers.AveragePooling2D(
      strides=(strides, strides))

  def call(self, x):
    x = tf.expand_dims(x, axis=-1)
    x = self._pooling(x)
    x = tf.squeeze(x, axis=-1)
    return x


class HopfieldModel(tf.keras.Sequential):

  def __init__(self, dt, hidden_units, max_delta_t, rel_tol,
               **kwargs):
    self.dt = dt
    self.hidden_units = hidden_units
    self.max_delta_t = max_delta_t
    self.rel_tol = rel_tol

    potential = tf.keras.Sequential([
        tf.keras.layers.Dense(hidden_units, activation='relu'),
        tf.keras.layers.Dense(1, activation=None),
        tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=-1))
    ])

    def get_kinetic(mass):
      return lambda x: 0.5 * mass * tf.reduce_mean(x * x, axis=-1)

    kinetic = get_kinetic(mass=1)

    def energy(x):
      return kinetic(x) + potential(x)

    stop_condition = StopCondition(energy, max_delta_t, rel_tol)

    layers = [
      tf.keras.Input([28, 28]),
      AveragePool2D(strides=2),
      tf.keras.layers.Reshape([14 * 14]),
      tf.keras.layers.LayerNormalization(),
      tf.keras.layers.Dropout(0.3),
      HopfieldLayer(energy, dt, stop_condition),
      tf.keras.layers.Dense(32, activation='relu'),
      tf.keras.layers.Dense(
        10, activation='softmax', name='Softmax'),
    ]
    super().__init__(layers, **kwargs)

    # `tf.keras.Model` will trace the variables within the layers'
    # `trainable_variables` attributes. The variables initialized
    # outside the layers will not be traced, e.g. those of the `potential`.
    #
    # Extra variables are appended into the property `trainable_variables`
    # of `tf.keras.Model` via the `_trainable_weights` attribute.
    self._trainable_weights += potential.trainable_variables


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


def get_benchmark_model(dataset, hidden_units):
  layers = [
    tf.keras.Input([28, 28]),
    AveragePool2D(strides=2),
    tf.keras.layers.Reshape([14 * 14]),
    tf.keras.layers.LayerNormalization(),
  ]
  layers += [
    tf.keras.layers.Dense(_, activation='relu') for _ in hidden_units]
  layers += [
    tf.keras.layers.Dense(10, activation='softmax'),
  ]
  model = tf.keras.Sequential(layers)
  model.compile(
    loss=tf.losses.CategoricalCrossentropy(),
    optimizer=tf.optimizers.Adam(1e-3),
  )
  model.fit(dataset)
  return model


def compare(x1, x2):
  threshold = np.median(np.abs(x1), axis=-1, keepdims=True)
  return np.where(np.abs(x1) > threshold, (x1 - x2) / x1, 0)


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = process_data(x_train, y_train)
x_test, y_test = process_data(x_test, y_test)

num_data = 1024
x_train = x_train[:num_data]
y_train = y_train[:num_data]

dataset = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
           .repeat(30)
           .batch(128))

benchmark_model = get_benchmark_model(
  # dataset, [14 * 14, 512, 14 * 14, 32])
  dataset, [32])
y0 = tf.argmax(y_train[:128], axis=-1)
noised_x_train = add_pixalwise_gaussian_noise(1.0, 0.3)(x_train)

y01 = benchmark_model(x_train[:128])
y02 = benchmark_model(noised_x_train[:128])
dy_benchmark = compare(y01, y02)

sub_benchmark_model = tf.keras.Sequential(benchmark_model.layers[:-2])
y11 = sub_benchmark_model(x_train[:128])
y12 = sub_benchmark_model(noised_x_train[:128])
dy_sub_benchmark = compare(y11, y12)

model = HopfieldModel(
  dt=1e-0, hidden_units=512, max_delta_t=100., rel_tol=1e-2)
model.compile(
  loss=tf.losses.CategoricalCrossentropy(),
  optimizer=tf.optimizers.Adam(1e-3),
)
print('start training')
# model.fit(dataset, verbose=2)

# use custom training loop for the convenience of doing experiments
for step, (X, y_true) in enumerate(dataset):
  with tf.GradientTape() as g:
    y_pred = model(X, training=True)
    loss = model.loss(y_true, y_pred)
    loss = tf.debugging.assert_all_finite(loss, '')
  grads = g.gradient(loss, model.trainable_variables)
  model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
  tf.print('step', step)
  tf.print('loss', loss)

y21 = model(x_train[:128])
y22 = model(noised_x_train[:128])
dy_model = compare(y21, y22)

sub_model = tf.keras.Sequential(model.layers[:5])

y31 = sub_model(x_train[:128])
y32 = sub_model(noised_x_train[:128])
dy_sub_model = compare(y31, y32)
