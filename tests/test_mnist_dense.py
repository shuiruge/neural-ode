r"""
This script tests `node.base.get_node_function` on MNIST dataset without using
`keras` model API.

Without using `keras` API, the RAM occupation is verily O(1). As

```
@tf.function
def train_one_step(x, y):
    with tf.GradientTape() as tape:
        outputs = model(x)
        loss = loss_fn(y, outputs)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


@tf.function
def train(dataset):
    step = 0
    for x, y in dataset:
        loss = train_one_step(x, y)
        tf.print('step', step, 'loss', loss)
        step += 1

dataset = ...
train(dataset)
```

However, without the `tf.function` decorator, the RAM grows from 410M to ~3G
as the `num_grids` parameter goes from `10` to `1000`.

Why so? Maybe because of the caching mechanism of Python, on which TF
optimizes.

So, make all things C++ (compliled by TF) is crucial! The efficiency increases
madly!
"""

import numpy as np
import tensorflow as tf
from node.core import get_node_function
from node.fix_grid import RKSolver
from node.utils.initializers import GlorotUniform


# for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


@tf.function
def normalize(x, axis=None):
    M = tf.reduce_max(x, axis, keepdims=True)
    m = tf.reduce_min(x, axis, keepdims=True)
    return (x - m) / (M - m + 1e-8)


class MyLayer(tf.keras.layers.Layer):

    def __init__(self, units, dt, num_grids, **kwargs):
        super().__init__(**kwargs)
        self.dt = dt
        self.num_grids = num_grids

        t0 = tf.constant(0.)
        self.tN = t0 + num_grids * dt

        self._model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu',
                                  kernel_initializer=GlorotUniform(1e-1)),
            tf.keras.layers.Dense(units, activation='relu',
                                  kernel_initializer=GlorotUniform(1e-1)),
        ])
        self._model.build([None, units])

        @tf.function
        def fn(t, x):
            z = self._model(x)
            with tf.GradientTape() as g:
                g.watch(x)
                r = normalize(x, axis=-1)
            return g.gradient(r, x, z)

        self._node_fn = get_node_function(
            RKSolver(self.dt), tf.constant(0.), fn)

    def call(self, x):
        y = self._node_fn(self.tN, x)
        return y


def process(X, y):
    X = X / 255.
    X = tf.reshape(X, [-1, 28 * 28])
    y = tf.one_hot(y, 10)
    return tf.cast(X, tf.float32), tf.cast(y, tf.float32)


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = process(x_train, y_train)
x_test, y_test = process(x_test, y_test)

model = tf.keras.Sequential([
    tf.keras.layers.Input([28 * 28]),
    tf.keras.layers.Dense(64, activation='relu'),
    MyLayer(64, dt=1e-1, num_grids=10),
    tf.keras.layers.Dense(10, activation='softmax')
])

optimizer = tf.compat.v1.train.AdamOptimizer()
loss_fn = tf.losses.CategoricalCrossentropy()


@tf.function
def train_one_step(x, y):
    with tf.GradientTape() as tape:
        outputs = model(x)
        loss = loss_fn(y, outputs)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


@tf.function
def train(dataset):
    step = 0
    for x, y in dataset:
        loss = train_one_step(x, y)
        tf.print('step', step, 'loss', loss)
        step += 1


num_epochs = 10
batch_size = 64

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.repeat(num_epochs).batch(batch_size)

# one approach
train(dataset)
# runs fast
# RAM 410M -> 410M as num_grids = 10 -> 1000

# other approach
# step = 0
# for x, y in dataset:
#     with tf.GradientTape() as tape:
#         outputs = model(x)
#         loss = loss_fn(y, outputs)
#     grads = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(grads, model.trainable_variables))
#     tf.print('step', step, 'loss', loss)
#     step += 1
# runs not as that fast
# RAM 410M -> ~3G as num_grids = 10 -> 1000
