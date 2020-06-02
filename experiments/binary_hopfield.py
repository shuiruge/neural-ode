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
DEBUG = False


class HopfieldLayer(tf.keras.layers.Layer):

  def __init__(self, energy, dt, beta=1.0, eps=1e-2,
               name='HopfieldLayer', **kwargs):
    super().__init__(name=name, **kwargs)

    self.energy = energy
    self.dt = tf.convert_to_tensor(dt)
    self.beta = tf.convert_to_tensor(beta)
    self.eps = eps
    self._config = {'energy': energy, 'dt': dt, 'beta': beta, 'eps': eps}

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
        e = energy(x)
      grad = g.gradient(e, x, unconnected_gradients='zero')
      # \sigma(x) := (1/2) (1 - cos(x))
      return tf.sqrt(x * (1 - x)) * grad

    max_delta_t = 50.0
    tolerance = 1e-2

    @tf.function
    def stop_condition(t0, x0, t, x):
      if DEBUG:
        tf.print('pvf:', tf.abs(pvf(t, x))[0])
        tf.print('energy:', energy(x)[:3])
      if tf.abs(t - t0) > max_delta_t:
        if DEBUG:
          tf.print('reached max delta t......')
        return True
      max_abs_velocity = tf.reduce_mean(
        tf.reduce_max(tf.abs(pvf(t, x)), axis=-1))
      if max_abs_velocity < tolerance:
        if DEBUG:
          tf.print('----------------- relaxed!!! ----------------')
        return True
      return False

    self.energy = energy
    self.pvf = pvf
    self._node_fn = get_dynamical_node_function(
      dynamical_solver, solver, pvf, stop_condition)

  @tf.function
  def call(self, x0):
    t0 = tf.constant(0.)
    x1 = self._node_fn(t0, x0)
    return x1

  def get_config(self):
    config = super().get_config()
    for k, v in self._config.items():
      config[k] = v
    return config


class HopfieldModel(tf.keras.Model):

  def __init__(self, units, dt, hidden_units=1024,
               **kwargs):
    super().__init__(**kwargs)
    self.units = units
    self.dt = dt
    self.hidden_units = hidden_units
    self._config = {'units': units, 'dt': dt, 'hidden_units': hidden_units}

    energy = tf.keras.Sequential([
        tf.keras.Input([units]),
        tf.keras.layers.Dense(hidden_units, activation='relu'),
        tf.keras.layers.Dense(1, activation=None)],
      name='Energy')

    self._down_sampling = tf.keras.layers.Dense(
      units, activation='sigmoid', name='DownSampling')
    self._hopfield_layer = HopfieldLayer(energy, dt)
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


mnist = tf.keras.datasets.mnist
(x_train, y_train), _ = mnist.load_data()
x_train, y_train = process_data(x_train, y_train)

scalar = StandardScaler()
scalar.fit(x_train)
x_train = scalar.transform(x_train)

# num_data = 1024
# x_train = x_train[:num_data]
# y_train = y_train[:num_data]

model = HopfieldModel(
  units=64, dt=1e-2, hidden_units=1024)
model.compile(
  loss=tf.losses.CategoricalCrossentropy(),
  optimizer=tfa.optimizers.Lookahead(tf.optimizers.Adam(1e-3)),
  # optimizer=tf.optimizers.Adamax(1e-3),
  metrics=['accuracy'],
)
print([(_.name, _.shape) for _ in model.trainable_variables])

# use custom training loop for the convenience of doing experiments
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

print('start training')

# TODO: where is the NaN?
for epoch in range(15):
  for step, (X, y_true) in enumerate(dataset.batch(32)):
    with tf.GradientTape() as g:
      y_pred = model(X)
      loss = model.loss(y_true, y_pred)
      loss = tf.debugging.assert_all_finite(loss, '')
    grads = g.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
    if step % 10 == 0:
      tf.print('step', step)
      tf.print('loss', loss)
model.evaluate(x_train, y_train, verbose=2)
