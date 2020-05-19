import numpy as np
import tensorflow as tf
from node.utils.layers import TracedModel, get_layerwise_gradients


tf.keras.backend.clear_session()
# tf.compat.v1.disable_eager_execution()


model = tf.keras.Sequential([
  tf.keras.layers.Dense(1024, activation='relu'),
  tf.keras.layers.Dense(10),
])
model = TracedModel(model)
model.compile(
  loss=tf.losses.CategoricalCrossentropy(from_logits=True),
  metrics=['acc'],
  # run_eagerly=True,
)

mnist = tf.keras.datasets.mnist
(x_train, y_train), _ = mnist.load_data()
x_train = x_train / 255.
x_train = np.reshape(x_train, [-1, 28 * 28])
y_train = np.eye(10)[y_train]

model.fit(x_train, y_train)

x = tf.constant(x_train[:32], dtype='float32')
t = tf.constant(y_train[:32], dtype='float32')

gradients = get_layerwise_gradients(model, x, t)
assert (model.trace[1][0] is model.trace[0][1])
assert (gradients[1][0] is not gradients[0][1])
