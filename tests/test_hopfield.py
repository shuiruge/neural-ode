import numpy as np
import tensorflow as tf
from node.hopfield import DiscreteTimeHopfieldLayer, ContinuousTimeHopfieldLayer


# for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

tf.keras.backend.clear_session()


IMAGE_SIZE = (16, 16)
MEMORY_SIZE = 50
FLIP_RATIO = 0.2
IS_CONTINUOUS_TIME = True
IS_BINARY = True
# LOSS = 'mae'
LOSS = 'binary_crossentropy'
EPOCHS = 500


def pooling(X, size):
  X = np.expand_dims(X, axis=-1)
  X = tf.image.resize(X, size).numpy()
  return X


def process_data(X, y, is_binary):
  X = X / 255
  X = pooling(X, IMAGE_SIZE)
  X = np.reshape(X, [-1, IMAGE_SIZE[0] * IMAGE_SIZE[1]])
  X = X * 2 - 1
  if is_binary:
    X = np.where(X < 0, -1, 1)
  y = np.eye(10)[y]
  return X.astype('float32'), y.astype('float32')


def train(hopfield, x_train, loss, epochs):
  # wraps the `hopfield` into a `tf.keras.Model` for training
  model = tf.keras.Sequential([
    hopfield,
  ])

  def rescale(x):
    """Rescales x from [-1, 1] to [0, 1]."""
    return x / 2 + 0.5

  def loss_fn(y_true, y_pred):
    y_true = rescale(y_true)
    y_pred = rescale(y_pred)
    return tf.reduce_mean(getattr(tf.losses, loss)(y_true, y_pred))

  optimizer = tf.optimizers.Adam(1e-3)
  model.compile(loss=loss_fn, optimizer=optimizer)
  model.fit(x_train, x_train, epochs=epochs, verbose=2)


def show_denoising_effect(hopfield, X):
  X = tf.convert_to_tensor(X)
  noised_X = tf.where(tf.random.uniform(shape=X.shape) < FLIP_RATIO,
                      -X, X)
  X_star = hopfield(noised_X)
  if isinstance(hopfield, ContinuousTimeHopfieldLayer):
    tf.print('relaxed at:', hopfield.stop_condition.relax_time)

  tf.print('(mean, var) of noised errors:',
           tf.nn.moments(tf.abs(noised_X - X), axes=[0, 1]))
  tf.print('(mean, var) of relaxed errors:',
           tf.nn.moments(tf.abs(X_star - X), axes=[0, 1]))
  tf.print('max of noised error:', tf.reduce_max(tf.abs(noised_X - X)))
  tf.print('max of relaxed error:', tf.reduce_max(tf.abs(X_star - X)))


def create_hopfield_layer():
  units = IMAGE_SIZE[0] * IMAGE_SIZE[1]
  if IS_CONTINUOUS_TIME:
    hopfield = ContinuousTimeHopfieldLayer(units)
  else:
    hopfield = DiscreteTimeHopfieldLayer(units)
  return hopfield


mnist = tf.keras.datasets.mnist
(x_train, y_train), _ = mnist.load_data()
x_train, y_train = process_data(x_train, y_train, IS_BINARY)
x_train = x_train[:MEMORY_SIZE]

hopfield = create_hopfield_layer()
train(hopfield, x_train, LOSS, EPOCHS)
show_denoising_effect(hopfield, x_train)
