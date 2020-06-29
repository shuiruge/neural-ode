r"""

Description
-----------

Implement the Hopfield network. Try to learn the Hebb rule (or better) using
modern SGD methods.

References
----------

1. Information Theory, Inference, and Learning Algorithms, chapter 42,
   Algorithm 42.9.

"""

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


# for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# limit CPU usage
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

tf.keras.backend.clear_session()

DEBUG = 1
IMAGE_SIZE = (8, 8)



def compute_w(kernel):
  w = (kernel + tf.transpose(kernel)) / 2
  w = tf.linalg.set_diag(w, tf.zeros(kernel.shape[0:-1]))
  return w


class HopfieldLayer(tf.keras.layers.Layer):

  def __init__(self, units, activation, name='HopfieldLayer', **kwargs):
    super().__init__(name=name, **kwargs)
    self.units = units
    self.activation = activation

  def build(self, batch_input_shape):
    self.kernel = self.add_weight(
        name='kernel', shape=[self.units, self.units],
        initializer='glorot_normal')

  def call(self, x):
    w = compute_w(self.kernel)
    a = tf.matmul(x, w)
    return self.activation(a)


def pooling(X, size):
  X = np.expand_dims(X, axis=-1)
  X = tf.image.resize(X, size).numpy()
  return X


def process_data(X, y):
  X = X / 255
  X = pooling(X, IMAGE_SIZE)
  X = np.reshape(X, [-1, IMAGE_SIZE[0] * IMAGE_SIZE[1]])
  X = np.where(X < 0.5, 0, 1)
  y = np.eye(10)[y]
  return X.astype('float32'), y.astype('float32')


mnist = tf.keras.datasets.mnist
(x_train, y_train), _ = mnist.load_data()
x_train, y_train = process_data(x_train, y_train)

hopfield = HopfieldLayer(64, tf.nn.sigmoid)
optimizer = tf.optimizers.Adam(1e-3)


def train_step(X):
  with tf.GradientTape() as g:
    next_X = hopfield(X)
    loss = tf.reduce_mean(tf.losses.binary_crossentropy(X, next_X))
  vars = hopfield.trainable_variables
  grads = g.gradient(loss, vars)
  optimizer.apply_gradients(zip(grads, vars))
  print(loss)


dataset = tf.data.Dataset.from_tensor_slices(x_train[:10])
dataset = dataset.shuffle(100).repeat(10000).batch(6)

for X in dataset:
  train_step(X)


