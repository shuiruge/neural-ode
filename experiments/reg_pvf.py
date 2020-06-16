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
DEBUG = 1


class NodeLayer(tf.keras.layers.Layer):

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
      tf.keras.layers.Dense(hidden_units, activation='relu'),
      tf.keras.layers.Dense(units)])

    self._node_fn = tf.function(
      get_node_function(self._solver, lambda t, x: self._fn(x)))

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
    tfa.layers.GroupNormalization(name='InputNormalization'),
    tf.keras.layers.Dense(units, name='FeatureExtractor'),
    tf.keras.layers.Dense(hidden_units, activation='relu'),
    tf.keras.layers.Dense(units),
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
    tfa.layers.GroupNormalization(name='InputNormalization'),
    tf.keras.layers.Dense(units, name='FeatureExtractor'),
    NodeLayer(0., t, 0.1, units, hidden_units, regular_factor),
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
           .batch(128))


units = 128
hidden_units = 512
benchmark_model = get_benchmark_model(dataset, units, hidden_units)

t = 2.
regular_factor = 1.
model = get_model(dataset, units, hidden_units, t, regular_factor)

noised_x_train = add_pixalwise_gaussian_noise(1, 0.3)(x_train)
y_pred_benchmark = benchmark_model.predict(noised_x_train)
y_pred = model.predict(noised_x_train)

def get_acc(y_true, y_pred):
  true_class = tf.argmax(y_true, axis=-1)
  pred_class = tf.argmax(y_pred, axis=-1)
  is_correct = tf.where(true_class == pred_class, 1., 0.)
  return tf.reduce_mean(is_correct)

print(get_acc(y_train, y_pred_benchmark))
print(get_acc(y_train, y_pred))

node_fn = model.layers[3]._node_fn
fn = model.layers[3]._fn
X = x_train[:32]

def get_pp(x):
  for layer in model.layers[:4]:
    x = layer(x)
  return x

def get_pvf(x):
  return model.layers[3]._fn(x)

def get_node_input(x):
  for layer in model.layers[:3]:
    x = layer(x)
  return x

x0 =  get_node_input(X)
xt = node_fn(tf.constant(0.), tf.constant(t), x0)
x1 = node_fn(tf.constant(0.), tf.constant(1.), x0)
x2 = node_fn(tf.constant(0.), tf.constant(2.), x0)
x10 = node_fn(tf.constant(0.), tf.constant(10.), x0)
pvf0 = get_pvf(x0)
pvft = get_pvf(xt)
pvf1 = get_pvf(x1)
pvf2 = get_pvf(x2)
pvf10 = get_pvf(x10)
