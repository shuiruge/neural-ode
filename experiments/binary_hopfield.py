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
from node.hopfield import hopfield
from node.utils.layers import (TracedModel, get_layer_activations,
                               get_layerwise_gradients, get_weights,
                               get_weight_gradiants, get_optimizer_variables)


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


class StopCondition(tf.Module):  # TODO: add `min_delta_t`.

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
    self._config = {'energy': energy, 'dt': dt,
                    'stop_condition': stop_condition}

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
    y = inverse_boundary_reflect(x1)
    return y

  def get_config(self):
    config = super().get_config()
    for k, v in self._config.items():
      config[k] = v
    return config


class HopfieldModel(tf.keras.Model):

  def __init__(self, units, dt, hidden_units, max_delta_t, rel_tol,
               **kwargs):
    super().__init__(**kwargs)
    self.units = units
    self.dt = dt
    self.hidden_units = hidden_units
    self.max_delta_t = max_delta_t
    self.rel_tol = rel_tol
    self._config = {'units': units, 'dt': dt, 'hidden_units': hidden_units}

    energy = tf.keras.Sequential([
        tf.keras.Input([units]),
        tf.keras.layers.Dense(hidden_units, activation='relu'),
        tf.keras.layers.Dense(1, activation=None)],
      name='Energy')

    self._down_sampling = tf.keras.layers.Dense(
      units, activation='sigmoid', name='DownSampling')
    stop_condition = StopCondition(energy, max_delta_t, rel_tol)
    self._hopfield_layer = HopfieldLayer(energy, dt, stop_condition)
    self._output_layer = tf.keras.layers.Dense(
      10, activation='softmax', name='Softmax')

  def call(self, x):
    x = self._down_sampling(x)
    x = tf.debugging.assert_all_finite(x, '')
    x = self._hopfield_layer(x)
    x = tf.debugging.assert_all_finite(x, '')
    x = self._output_layer(x)
    x = tf.debugging.assert_all_finite(x, '')
    return x

  def get_config(self):
    config = super().get_config()
    for k, v in self._config.items():
      config[k] = v
    return config


def process_data(X, y):
  X = X / 255.
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


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = process_data(x_train, y_train)
x_test, y_test = process_data(x_test, y_test)

scalar = StandardScaler()
scalar.fit(x_train)
x_train = scalar.transform(x_train)
x_test = scalar.transform(x_test)

# num_data = 1024
# x_train = x_train[:num_data]
# y_train = y_train[:num_data]

model = HopfieldModel(
  units=64, dt=1e-2, hidden_units=512, max_delta_t=100., rel_tol=5e-3)
model.compile(
  loss=tf.losses.CategoricalCrossentropy(),
  optimizer=tfa.optimizers.Lookahead(tf.optimizers.Adam(1e-3)),
  # optimizer=tf.optimizers.Adamax(1e-3),
  metrics=['accuracy'],
)

# use custom training loop for the convenience of doing experiments
dataset = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
           .repeat(2)
           .batch(128))

print('start training')

# TODO: where is the NaN?
for step, (X, y_true) in enumerate(dataset):
  with tf.GradientTape() as g:
    y_pred = model(X)
    loss = model.loss(y_true, y_pred)
    loss = tf.debugging.assert_all_finite(loss, '')
  grads = g.gradient(loss, model.trainable_variables)
  model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
  tf.print('step', step)
  tf.print('loss', loss)

noised_x_train = add_random_flip_noise(scalar, 0.01)(x_train)
y0 = tf.argmax(model(noised_x_train[:128]), axis=-1)
y1 = tf.argmax(y_train[:128], axis=-1)

model2 = HopfieldModel(
  units=64, dt=1e-2, hidden_units=512, max_delta_t=100., rel_tol=3e-3)
model2.compile(
  loss=tf.losses.CategoricalCrossentropy(),
  optimizer=tfa.optimizers.Lookahead(tf.optimizers.Adam(1e-3)),
  metrics=['accuracy'],
)
model2(X)
model2.set_weights(model.get_weights())

y10 = tf.argmax(model2(noised_x_train[:128]), axis=-1)
