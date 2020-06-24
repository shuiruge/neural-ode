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

# limit CPU usage
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

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
    if DEBUG == 2:
      tf.print('energy:', current_energy)
    if tf.abs(t - t0) > self.max_delta_t:
      shall_stop = True
      if DEBUG > 0:
        tf.print('reached max delta t:', self.max_delta_t)
    elif t > t0:
      delta_energy = current_energy - self._previous_energy
      rel_delta = (tf.math.abs(delta_energy) /
                   (tf.math.abs(self._previous_energy) / 2 +
                       tf.math.abs(current_energy) / 2 + 1e-8))
      if DEBUG == 2:
        tf.print('energy relative difference:', rel_delta)
      # max_rel_delta = tf.reduce_max(rel_delta)
      mean, var = tf.nn.moments(rel_delta, axes=[0])
      max_rel_delta = mean + 1 * tf.sqrt(var)
      if DEBUG == 2:
        tf.print('top energy relative difference:',
                 tf.math.top_k(rel_delta, k=5)[0],
                 '\n')
      shall_stop = max_rel_delta < self.rel_tol
      if shall_stop and DEBUG > 0:
        tf.print('relaxed at delta t:', t - t0)
    else:
      pass
    if DEBUG == 5:
      x_mean, x_var = tf.nn.moments(x, axes=[1])
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
      solver = RKF56Solver(self.dt, tol=1e-3, min_dt=1e-1)
      dynamical_solver = DynamicalRKF56Solver(self.dt, tol=1e-3, min_dt=1e-1)
    else:
      raise ValueError()

    def bound_fn(x):
      y = inverse_boundary_reflect(x)
      return y * 2 - 1

    def pvf(_, x):
      with tf.GradientTape() as g:
        g.watch(x)
        bounded_x = bound_fn(x)
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


class HopfieldModel(tf.keras.Sequential):

  def __init__(self, dt, d_model, hidden_units, max_delta_t, rel_tol,
               dropout_rate, **kwargs):
    self.dt = dt
    self.d_model = d_model
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

    kinetic = get_kinetic(mass=0)

    def energy(x):
      return kinetic(x) + potential(x)

    stop_condition = StopCondition(energy, max_delta_t, rel_tol)

    layers = [
      tf.keras.Input([28 * 28]),
      tf.keras.layers.LayerNormalization(name='InputLayerNorm'),
      tf.keras.layers.Dense(d_model, name='DownSampling'),
      tf.keras.layers.LayerNormalization(name='DownSamplingLayerNorm'),
      tf.keras.layers.Dropout(dropout_rate),
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
  X = np.reshape(X, [-1, 28 * 28])
  y = np.eye(10)[y]
  return X.astype('float32'), y.astype('float32')


def get_benchmark_model(dataset, d_model, hidden_units):
  layers = [
    tf.keras.Input([28 * 28]),
    tf.keras.layers.LayerNormalization(name='InputLayerNorm'),
    tf.keras.layers.Dense(d_model, name='DownSampling'),
    tf.keras.layers.LayerNormalization(name='DownSamplingLayerNorm'),
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


def compare(x1, x2):
  threshold = np.median(np.abs(x1), axis=-1, keepdims=True)
  return np.where(np.abs(x1) > threshold, (x1 - x2) / (x1 + x2) * 2, 0)


mnist = tf.keras.datasets.mnist
(x_train, y_train), _ = mnist.load_data()
x_train, y_train = process_data(x_train, y_train)

# XXX: test!
# num_data = 1000
# x_train = x_train[:num_data]
# y_train = y_train[:num_data]

# use custom training loop for the convenience of doing experiments
dataset = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
           .shuffle(1000)
           .repeat(3)
           .batch(64))

model = HopfieldModel(
  dt=1e-0, d_model=32, hidden_units=128, max_delta_t=100., rel_tol=1e-2,
  dropout_rate=0.3)
model.compile(
  loss=tf.losses.CategoricalCrossentropy(),
  optimizer=tf.optimizers.Adam(1e-3),
)

print('start training')


@tf.function
def train_step(X, y_true):
  with tf.GradientTape() as g:
    y_pred = model(X, training=True)
    loss = model.loss(y_true, y_pred)
  grads = g.gradient(loss, model.trainable_variables)
  model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
  tf.print('loss', loss)


for step, (X, y_true) in enumerate(dataset):
  tf.print('step', step)
  train_step(X, y_true)

benchmark_model = get_benchmark_model(dataset, d_model=32, hidden_units=[32])

noised_x_train = add_pixalwise_gaussian_noise(0.5, 0.5)(x_train)

y01 = benchmark_model(x_train[:128])
y02 = benchmark_model(noised_x_train[:128])
dy_benchmark = compare(y01, y02)

sub_benchmark_model = tf.keras.Sequential(benchmark_model.layers[:-2])
y11 = sub_benchmark_model(x_train[:128])
y12 = sub_benchmark_model(noised_x_train[:128])
dy_sub_benchmark = compare(y11, y12)

y21 = model(x_train[:128])
y22 = model(noised_x_train[:128])
dy_model = compare(y21, y22)

sub_model = tf.keras.Sequential(model.layers[:-2])

y31 = sub_model(x_train[:128])
y32 = sub_model(noised_x_train[:128])
dy_sub_model = compare(y31, y32)

sub_model_raw = tf.keras.Sequential(model.layers[:-3])
y41 = sub_model_raw(x_train[:128])
y42 = sub_model_raw(noised_x_train[:128])
dy_sub_model_raw = compare(y42, y42)
