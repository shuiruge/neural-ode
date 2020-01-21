import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from node.core import get_node_function
from node.fix_grid import RKSolver
from node.utils.trajectory import tracer
from node.energy_based import Energy, energy_based, identity


# for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


DTYPE = 'float32'
tf.keras.backend.set_floatx(DTYPE)


class MyLayer(tf.keras.layers.Layer):

    def __init__(self, units, dt, num_grids, **kwargs):
        super().__init__(**kwargs)
        self.dt = dt
        self.num_grids = num_grids

        t0 = tf.constant(0., dtype=DTYPE)
        self.tN = t0 + num_grids * dt

        self._model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu', dtype=DTYPE),
            tf.keras.layers.Dense(units, dtype=DTYPE),
        ])
        self._model.build([None, units])

        self._raw_pvf = lambda _, x: self._model(x)
        self._energy = Energy(identity, self._raw_pvf)
        self._pvf = energy_based(identity, self._energy)
        self._node_fn = get_node_function(RKSolver(self.dt, dtype=DTYPE),
                                          tf.constant(0., dtype=DTYPE),
                                          self._pvf)

    def call(self, x):
        y = self._node_fn(self.tN, x)
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

model = tf.keras.Sequential([
    tf.keras.layers.Input([28 * 28]),
    tf.keras.layers.Dense(64, use_bias=False),  # down-sampling
    MyLayer(64, dt=1e-1, num_grids=10),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer=tf.optimizers.Nadam(1e-3, epsilon=1e-2),
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=12, batch_size=128)

my_layer_id = 1
my_layer = model.layers[my_layer_id]
trace = tracer(RKSolver(0.1), my_layer._pvf)
energy = Energy(identity, my_layer._model)
truncated_model = tf.keras.Sequential(model.layers[:my_layer_id])

hidden = truncated_model(x_train[:100])
labels = y_train[:100]
trajectory = trace(t0=tf.constant(0.),
                   t1=tf.constant(5.),
                   dt=tf.constant(0.1),
                   x=hidden)
mean, variance = tf.nn.moments(trajectory, axes=[-1])
tf.print('trajectory[0]\n', trajectory[0], '\n')
tf.print('variance[0]\n', variance[0], '\n')
