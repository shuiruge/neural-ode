r"""

Description
-----------

This script tries to reproduce the arguments in ref [1].

Experiment Results
------------------

It is find that the argument in ref [1], i.e. TisODE is more robust, is in
dependent in the case of deep feed forward network. In fact, the accuracy of
TisODE decreases much faster than the benchmark model when adding the same
Gaussian noise.

Adding `GroupNormalization` (or `LayerNormalization`) layer will increases
the robustness, making TisODE compatible with the vanilla, but not significantly
surpassed. Even, evaluating the ODE for a longer period will decrease the
accuracy, indicating that the fixed point is not fixed.

Chaning ODESolver from RK4 to RKF56 improves the performance, but also gains
more temporal occupation. The qualitative conclusion is invariant.

Analysis
--------

The regularization trick in ref [1] is suspicous for ensuring robustness,
since the fixed point is not ensured to be stable. An unstable fixed point
can also provide a small enough regularization term.

References
----------

1. arXiv: 1910.05513

"""

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


SOLVER = 'rk4'
# SOLVER = 'rkf56'


class NodeLayer(tf.keras.layers.Layer):
  r"""Implements the TisODE in ref [1].

  References:
    1. arXiv: 1910.05513

  Args:
    t0: float
      Being `0` in ref [1].
    t1: float
      The `2T` in ref [1].
    dt: float
      The :math:`\Delta t` parameter in the Runge-Kutta solver.
    units: int
      The representing dimension.
    hidden_units: int
    regular_factor: float
    n: int
      To compute the integral :math:`\int_{T}^{2T} dt \| f(x(t)) \|`, we
      seperate `x(t)` into `n` pieces, and then compute the Monte-Carlo
      integral instead.
  """

  def __init__(self, t0, t1, dt, units, hidden_units, regular_factor, n=5,
               name='NodeLayer', **kwargs):
    super().__init__(name=name, **kwargs)
    self.t0 = tf.convert_to_tensor(t0)
    self.t1 = tf.convert_to_tensor(t1)
    self.dt = tf.convert_to_tensor(dt)
    self.units = units
    self.hidden_units = hidden_units
    self.regular_factor = regular_factor
    self.n = n

    if SOLVER == 'rk4':
      self._solver = RK4Solver(self.dt)
    elif SOLVER == 'rkf56':
      self._solver = RKF56Solver(self.dt, tol=1e-2, min_dt=1e-1)
    else:
      raise ValueError()

    self._fn = tf.keras.Sequential([
      tf.keras.Input([units]),
      tf.keras.layers.Dense(hidden_units),
      tf.keras.layers.LayerNormalization(),
      tf.keras.layers.Lambda(tf.keras.activations.relu),
      tf.keras.layers.Dense(units)])

    self._node_fn = get_node_function(self._solver, lambda t, x: self._fn(x))

  def call(self, x0):
    t0 = self.t0
    t1 = self.t1 / 2
    dt = (t1 - t0) / self.n
    xs = tf.TensorArray(dtype=x0.dtype, size=self.n)
    x = self._node_fn(t0, t1, x0)
    for i in range(self.n):
      t0 = t1
      t1 = t1 + dt
      x = self._node_fn(t0, t1, x)
      xs = xs.write(i, x)

    # add regularization for ensuring stability
    pvfs = self._fn(tf.reshape(xs.stack(), [-1, self.units]))
    pvf_norms = tf.reduce_sum(
      tf.abs(pvfs) * tf.nn.softmax(tf.abs(pvfs)), axis=-1)
    regularization = self.regular_factor * tf.reduce_mean(pvf_norms)
    self.add_loss(regularization, inputs=True)

    return x


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
    tf.keras.layers.LayerNormalization(name='InputNormalization'),

    # Feature extractor
    tf.keras.layers.Dense(units, name='FeatureExtractor'),
    tf.keras.layers.LayerNormalization(name='FeatureNormalization'),

    # Representation mapping
    # Minic the structure in the TisODE model
    tf.keras.layers.Dense(hidden_units),
    tf.keras.layers.LayerNormalization(),
    tf.keras.layers.Lambda(tf.keras.activations.relu),
    tf.keras.layers.Dense(units),

    # Output classifier
    tf.keras.layers.Dense(10, activation='softmax',
                          name='OutputClassifier'),
  ])
  model.compile(
    loss=tf.losses.CategoricalCrossentropy(),
    optimizer=tfa.optimizers.Lookahead(tf.optimizers.Adam(1e-3)),
    metrics=['accuracy'],
  )
  model.fit(dataset, verbose=2, epochs=15)
  return model


def get_model(dataset, units, hidden_units, t, regular_factor):
  model = tf.keras.Sequential([
    tf.keras.layers.Reshape([28 * 28]),
    tf.keras.layers.LayerNormalization(name='InputNormalization'),

    # Feature extractor
    tf.keras.layers.Dense(units, name='FeatureExtractor'),
    tf.keras.layers.LayerNormalization(name='FeatureNormalization'),

    # Representation mapping
    NodeLayer(0., t, 0.1, units, hidden_units, regular_factor),

    # Output classifier
    tf.keras.layers.Dense(10, activation='softmax',
                          name='OutputClassifier'),
  ])
  model.compile(
    loss=tf.losses.CategoricalCrossentropy(),
    optimizer=tfa.optimizers.Lookahead(tf.optimizers.Adam(1e-3)),
    metrics=['accuracy'],
  )
  if dataset is not None:
    model.fit(dataset, verbose=2, epochs=15)
  return model


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = process_data(x_train, y_train)
x_test, y_test = process_data(x_test, y_test)

dataset = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
           .shuffle(10000)
           .batch(128))


units = 128
hidden_units = 512
benchmark_model = get_benchmark_model(dataset, units, hidden_units)

t = 2.
regular_factor = 1.
model = get_model(dataset, units, hidden_units, t, regular_factor)

noised_x_train = add_pixalwise_gaussian_noise(0.5, 0.5)(x_train)
y_pred_benchmark = benchmark_model.predict(noised_x_train)
y_pred = model.predict(noised_x_train)

def get_accuracy(y_true, y_pred):
  true_class = tf.argmax(y_true, axis=-1)
  pred_class = tf.argmax(y_pred, axis=-1)
  is_correct = tf.where(true_class == pred_class, 1., 0.)
  return tf.reduce_mean(is_correct)

print('Noised accuracy for benchmark model:',
      get_accuracy(y_train, y_pred_benchmark))
print('Noised accuracy for TisODE model:',
      get_accuracy(y_train, y_pred))

def fork_model(dataset, units, hidden_units, t, regular_factor, model):
  new_model = get_model(None, units, hidden_units, t, regular_factor)
  new_model.build((None, 28, 28))
  new_model.set_weights(model.get_weights())
  return new_model

model_2 = fork_model(dataset, units, hidden_units, 4., regular_factor, model)
y_pred_2 = model_2.predict(noised_x_train)
print('Noised accuracy for longer TisODE model:',
      get_accuracy(y_train, y_pred_2))

