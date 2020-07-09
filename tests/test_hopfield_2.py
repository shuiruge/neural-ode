import numpy as np
import tensorflow as tf

from node.hopfield import (ContinuousTimeHopfieldLayer,
                           DiscreteTimeHopfieldLayer)

# for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

tf.keras.backend.clear_session()


IMAGE_SIZE = (16, 16)
FLIP_RATIO = 0.1


def pooling(X, size):
  X = tf.expand_dims(X, axis=-1)
  X = tf.image.resize(X, size)
  return X


def process_data(X, y, image_size):
  X = pooling(X, image_size)
  X = X / 255.
  X = tf.where(X < 0.5, -1., 1.)
  X = tf.reshape(X, [-1, image_size[0] * image_size[1]])
  y = tf.one_hot(y, 10)
  return tf.cast(X, tf.float32), tf.cast(y, tf.float32)


def get_benchmark_model(model):
  layers = [layer for layer in model.layers
            if not isinstance(layer, ContinuousTimeHopfieldLayer)]
  return tf.keras.Sequential(layers)


model = tf.keras.Sequential([
  tf.keras.Input([IMAGE_SIZE[0] * IMAGE_SIZE[1]]),
  tf.keras.layers.LayerNormalization(),
  tf.keras.layers.Dense(512, activation='tanh'),
  ContinuousTimeHopfieldLayer(reg_factor=3, relax_tol=1e-2),
  tf.keras.layers.Dense(128, activation='tanh'),
  ContinuousTimeHopfieldLayer(reg_factor=3, relax_tol=1e-2),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax'),
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

mnist = tf.keras.datasets.mnist
(x_train, y_train), _ = mnist.load_data()
x_train, y_train = process_data(x_train, y_train, IMAGE_SIZE)

model.fit(x_train, y_train, epochs=50, verbose=2)


# ------- noise effect --------


benchmark_model = get_benchmark_model(model)

X = tf.convert_to_tensor(x_train[:100])
targets = tf.argmax(tf.convert_to_tensor(y_train[:100]), axis=1)
noised_X = tf.where(tf.random.uniform(shape=X.shape) < FLIP_RATIO,
                    -X, X)
unoised_y = tf.argmax(model.predict(X), axis=1)
y = tf.argmax(model.predict(noised_X), axis=1)
yb = tf.argmax(benchmark_model.predict(noised_X), axis=1)

sub_model = tf.keras.Sequential(model.layers[:5])
sy1 = sub_model.predict(X)
sy2 = sub_model.predict(noised_X)

ssub_model = tf.keras.Sequential(model.layers[:4])
ssy1 = ssub_model.predict(X)
ssy2 = ssub_model.predict(noised_X)

def get_error_ratio(original, noised, threshold=0.2):
  diff = original - noised
  ratio_per_sample = tf.reduce_mean(
    tf.cast(tf.abs(diff) > threshold, tf.float32), axis=1)
  return ratio_per_sample

num_misleading = 0
num_corrected = 0
num_uncorrected = 0
for err0, err1, ybi, yi, uyi, ti in zip(
    get_error_ratio(ssy1, ssy2).numpy(),
    get_error_ratio(sy1, sy2).numpy(),
    yb, y, unoised_y, targets):
  if yi == uyi and ybi != yi:
    num_corrected += 1
  elif ybi == uyi and ybi != yi:
    num_misleading += 1
  elif yi != uyi and ybi == yi:
    num_uncorrected += 1
  else:
    pass
  print(f'{err0:.3f} => {err1:.3f} | {ybi} => {yi} | {uyi} ({ti})')
print(f'misleading: {num_misleading}',
      f'corrected: {num_corrected}',
      f'uncorrected: {num_uncorrected}')
