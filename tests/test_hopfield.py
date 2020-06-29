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


def pooling(X, size):
  X = np.expand_dims(X, axis=-1)
  X = tf.image.resize(X, size).numpy()
  return X


def process_data(X, y):
  X = X / 255
  X = pooling(X, IMAGE_SIZE)
  X = np.reshape(X, [-1, IMAGE_SIZE[0] * IMAGE_SIZE[1]])
  X = np.where(X < 0.5, -1, 1)
  y = np.eye(10)[y]
  return X.astype('float32'), y.astype('float32')


def train(hopfield, x_train, epochs=500):
  model = tf.keras.Sequential([
    hopfield,
    tf.keras.layers.Lambda(lambda x: x / 2 + 0.5),
  ])
  model.compile(loss='binary_crossentropy', optimizer='adam')
  y_train = x_train / 2 + 0.5
  model.fit(x_train, y_train, epochs=epochs, verbose=2)


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
x_train, y_train = process_data(x_train, y_train)
x_train = x_train[:MEMORY_SIZE]

hopfield = create_hopfield_layer()
train(hopfield, x_train)
show_denoising_effect(hopfield, x_train)
