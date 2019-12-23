r"""
This script tests `node.base.get_node_function` on MNIST dataset without using
`keras` model API.

Without using `keras` API, the RAM occupation is verily O(1). As

```
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

However, without the `tf.function` decorator, the RAM grows from 380M to 3G
as the `num_grids` parameter goes from `10` to `1000`.

Why so? Maybe because of the caching mechanism of Python, on which TF
optimizes.
"""

import numpy as np
import tensorflow as tf
from node.base import get_node_function
from node.fix_grid import RKSolver


# for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class MyLayer(tf.keras.layers.Layer):

    def __init__(self, dt, num_grids, **kwargs):
        super().__init__(**kwargs)
        self.dt = dt
        self.num_grids = num_grids

        t0 = tf.constant(0.)
        self.tN = t0 + num_grids * dt

        input_dim = 28 * 28
        self._model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(input_dim, activation='relu'),
        ])
        self._model.build([None, input_dim])

        def fn(t, x):
            return self._model(x)

        self._node_fn = get_node_function(RKSolver(self.dt), 0., fn)

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
    MyLayer(dt=1e-2, num_grids=10),
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
train(dataset)
