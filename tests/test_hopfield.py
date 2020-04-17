import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from node.core import get_node_function
from node.solvers.runge_kutta import RK4Solver
from node.hopfield import hopfield, identity


# for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


DTYPE = 'float32'
tf.keras.backend.set_floatx(DTYPE)

class HopfieldLayer(tf.keras.layers.Layer):

    def __init__(self, units, dt, t,
                 lower_bounded_fn,
                 linear_transform=identity,
                 **kwargs):
        super().__init__(**kwargs)
        self.dt = tf.convert_to_tensor(dt, dtype=DTYPE)
        self.t = tf.convert_to_tensor(t, dtype=DTYPE)
        self.lower_bounded_fn = lower_bounded_fn
        self.linear_transform = linear_transform

        t0 = tf.constant(0.)

        self._model = tf.keras.Sequential([
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(units),
        ])
        self._model.build([None, units])
        self._pvf = hopfield(
            lambda x: self.lower_bounded_fn(self._model(x)),
            self.linear_transform)
        solver = RK4Solver(self.dt)
        self._node_fn = get_node_function(
            solver, tf.constant(0.), self._pvf)

    def call(self, x):
        y = self._node_fn(self.t, x)
        return y

    def get_config(self):
        return super().get_config().copy()


def process(X, y):
  X = X / 255.
  X = np.reshape(X, [-1, 28 * 28])
  y = np.eye(10)[y]
  return X.astype(DTYPE), y.astype(DTYPE)


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = process(x_train, y_train)
x_test, y_test = process(x_test, y_test)

scalar = StandardScaler()
scalar.fit(x_train)
x_train = scalar.transform(x_train)
x_test = scalar.transform(x_test)


def get_compiled_model(lower_bounded_fn, t, lr=1e-3, epsilon=1e-3):
  model = tf.keras.Sequential([
    tf.keras.layers.Input([28 * 28]),
    tf.keras.layers.Dense(64, use_bias=False),  # down-sampling
    tf.keras.layers.LayerNormalization(),
    HopfieldLayer(64, dt=1e-1, t=t,
                  linear_transform=identity,
                  lower_bounded_fn=lower_bounded_fn),
    tf.keras.layers.Dense(10, activation='softmax')
  ])

  model.compile(
      loss='categorical_crossentropy',
      optimizer=tf.optimizers.Adam(lr, epsilon=epsilon),
      metrics=['accuracy'])

  return model


@tf.function
def lower_bounded_fn(x):
    return 8 * tf.sqrt(tf.reduce_sum(tf.square(x), axis=1))

model = get_compiled_model(lower_bounded_fn, t=1.)
model.fit(x_train, y_train, epochs=10, batch_size=128)
score = model.evaluate(x_train, y_train)

longer_period_model = get_compiled_model(lower_bounded_fn, t=3.)
longer_period_model.set_weights(model.get_weights())
longer_period_score = longer_period_model.evaluate(x_train, y_train)

for i in range(2):
  print(f'{score[i]:.5f} => {longer_period_score[i]:.5f}')
