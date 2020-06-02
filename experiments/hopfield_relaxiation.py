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


_LOG = []


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
      # return lower_bounded_fn(fn(x))
      return tf.reduce_mean(tf.square(x) + tf.tanh(fn(x)), axis=-1)

    def pvf(_, x):
      with tf.GradientTape() as g:
        g.watch(x)
        e = energy(x)
      return -g.gradient(e, x, unconnected_gradients='zero')

    # boosted_pvf = rmsprop(pvf, eps=1e-3)

    max_delta_t = 50.0
    tolerance = 5e-2

    @tf.function
    def stop_condition(t0, x0, t, x):
      # tf.print(tf.math.top_k(tf.abs(pvf(t, x)), k=5)[0][:3])
      tf.print('pvf:', tf.abs(pvf(t, x))[0])
      tf.print('energy:', energy(x)[:3])
      if tf.abs(t - t0) > max_delta_t:
        tf.print('reached max delta t......')
        return True
      # max_abs_velocity = tf.reduce_mean(tf.abs(pvf(t, x)))
      max_abs_velocity = tf.reduce_mean(tf.reduce_max(tf.abs(pvf(t, x)), axis=-1))
      if max_abs_velocity < tolerance:
        tf.print('----------------- relaxed!!! ----------------')
        return True
      return False

    self.energy = energy
    self.pvf = pvf
    # self.boosted_pvf = boosted_pvf
    self._node_fn = get_dynamical_node_function(
      dynamical_solver, solver, pvf, stop_condition)

  @tf.function
  def call(self, x0):
    t0 = tf.constant(0.)

    # ms = nest_map(lambda x: tf.ones_like(x))(x0)
    # x0 = (x0, ms)

    y = self._node_fn(t0, x0)

    # y = y[0]
    tf.print(y[:3])
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
    layers.append(
      tf.keras.layers.LayerNormalization(name='HopfieldLayerNorm'))

    layers += [
      tf.keras.layers.Dense(128, name='Hidden'),
      tf.keras.layers.Dense(10, name='OutputLogits'),
    ]

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
                    # np.zeros_like(binary),
                    np.ones_like(binary),
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

print('start training')

for epoch in range(15):
  for step, (X, y_true) in enumerate(dataset.batch(32)):
    tf.print('step', step)
    with tf.GradientTape() as g:
      y_pred = model(X)
      loss = model.loss(y_true, y_pred)
    grads = g.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
    tf.print('loss', loss)
  model.evaluate(x_train, y_train, verbose=2)

noised_x_train = add_random_flip_noise(scalar, 0.1)(x_train)
y_pred = model(noised_x_train)
y_pred = tf.argmax(tf.nn.softmax(y_pred, axis=-1), axis=-1)
y_true = tf.argmax(y_train, axis=-1)


