import numpy as np
import tensorflow as tf

from node.hopfield import (ContinuousTimeHopfieldLayer,
                           DiscreteTimeHopfieldLayer)

# for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

tf.keras.backend.clear_session()


# IMAGE_SIZE = (32, 32)
# MEMORY_SIZE = 500
IMAGE_SIZE = (16, 16)
MEMORY_SIZE = 50
FLIP_RATIO = 0.2
IS_BENCHMARK = False
USE_HEBB_RULE_INITIALIZER = False
IS_CONTINUOUS_TIME = True
DATA_QUANTIZE_METHOD = 'binary'
# DATA_QUANTIZE_METHOD = 'four-piece'
EPOCHS = 500


def pooling(X, size):
  X = np.expand_dims(X, axis=-1)
  X = tf.image.resize(X, size).numpy()
  return X


def process_data(X, image_size, data_quantize_method, memory_size):
  X = pooling(X, image_size)
  X = np.reshape(X, [-1, image_size[0] * image_size[1]])
  X = X / 255 * 2 - 1
  if data_quantize_method == 'binary':
    X = np.where(X < 0, -1, 1)
  elif data_quantize_method == 'four-piece':
    X = np.where(X > 0.5, 1,
                 np.where(X > 0, 0.5,
                          np.where(X > -0.5, -0.5, -1)))
  else:
    pass
  X = X[:memory_size]
  return X.astype('float32')


def hebb_rule_initializer(X):
  n = X.shape[-1]
  w = np.matmul(np.transpose(X), X)
  w = w / np.std(w) / n  # normalize to w ~ N(0, 1/n) (Xavi initializer).
  b = np.zeros([n])
  return w, b


def train(hopfield, x_train, epochs, use_hebb_rule_initializer):
  # wraps the `hopfield` into a `tf.keras.Model` for training
  model = tf.keras.Sequential([
    tf.keras.Input([x_train.shape[-1]]),
    hopfield,
  ])

  def rescale(x):
    """Rescales x from [-1, 1] to [0, 1]."""
    return x / 2 + 0.5

  optimizer = tf.optimizers.Adam(1e-3)
  model.compile(optimizer=optimizer,
                loss=lambda *args: tf.constant(0.))
  if use_hebb_rule_initializer:
    model.set_weights(hebb_rule_initializer(x_train))
  model.fit(x_train, x_train, epochs=epochs, verbose=2)


def show_denoising_effect(hopfield, X, flip_ratio):
  X = tf.convert_to_tensor(X)
  noised_X = tf.where(tf.random.uniform(shape=X.shape) < flip_ratio,
                      -X, X)
  X_star = hopfield(noised_X)
  if isinstance(hopfield, ContinuousTimeHopfieldLayer):
    tf.print('relaxed at:', hopfield._stop_condition.relax_time)

  tf.print('(mean, var) of noised errors:',
           tf.nn.moments(tf.abs(noised_X - X), axes=[0, 1]))
  tf.print('(mean, var) of relaxed errors:',
           tf.nn.moments(tf.abs(X_star - X), axes=[0, 1]))
  tf.print('max of noised error:', tf.reduce_max(tf.abs(noised_X - X)))
  tf.print('max of relaxed error:', tf.reduce_max(tf.abs(X_star - X)))


def create_hopfield_layer(units, is_continuous_time):
  if is_continuous_time:
    hopfield = ContinuousTimeHopfieldLayer(reg_factor=1)
  else:
    hopfield = DiscreteTimeHopfieldLayer(reg_factor=1)
  return hopfield


mnist = tf.keras.datasets.mnist
(x_train, _), _ = mnist.load_data()
x_train = process_data(x_train, IMAGE_SIZE, DATA_QUANTIZE_METHOD, MEMORY_SIZE)

hopfield = create_hopfield_layer(IMAGE_SIZE[0] * IMAGE_SIZE[1],
                                 IS_CONTINUOUS_TIME)
if IS_BENCHMARK:
  train(hopfield, x_train, 0, True)
else:
  train(hopfield, x_train, EPOCHS, USE_HEBB_RULE_INITIALIZER)
show_denoising_effect(hopfield, x_train, FLIP_RATIO)
