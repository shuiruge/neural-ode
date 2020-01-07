import numpy as np
import tensorflow as tf
from tensorflow.python.framework.errors_impl import NotFoundError
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
    """convolution + normalization"""

    def __init__(self, filters, kernel_size, solver, t, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.solver = solver
        self.t = t

        self.convolve = tf.keras.layers.Conv2D(
            filters, kernel_size, activation='relu', padding='same',
            kernel_initializer=GlorotUniform(1e-1))

        @tf.function
        def pvf(t, x):
            z = self.convolve(x)
            with tf.GradientTape() as g:
                g.watch(x)
                r = normalize(x, axis=[-3, -2])
            return g.gradient(r, x, z)

        self._pvf = pvf
        self._node_fn = get_node_function(solver, 0., pvf)

    def call(self, x):
        y = self._node_fn(self.t, x)
        return y


def get_compiled_model(num_filters, kernel_size, t, save_path=None):
    model = tf.keras.Sequential([
        tf.keras.layers.Input([28, 28]),
        tf.keras.layers.Reshape([28, 28, 1]),

        # Down-sampling
        tf.keras.layers.Conv2D(num_filters, kernel_size),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(),

        MyLayer(num_filters, kernel_size, RKSolver(0.1), t),

        # Classification
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax'),
    ])
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    if save_path is not None:
        try:
            model.load_weights(save_path)
        except NotFoundError as e:
            print(str(e))

    return model


def main(num_filters, kernel_size, save_path,
         train_t=None, test_ts=None):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if train_t:
        model = get_compiled_model(
            num_filters, kernel_size, train_t, save_path)
        model.summary()
        model.fit(x_train, y_train, epochs=100,
                  validation_data=(x_test, y_test),
                  callbacks=[tf.keras.callbacks.EarlyStopping()])
        model.save_weights(save_path)

    elif test_ts:
        for t in test_ts:
            model = get_compiled_model(
                num_filters, kernel_size, t, save_path)
            model.summary()
            model.load_weights(save_path)
            print(f'evaluate at t = {t}:')
            model.evaluate(x_test, y_test, verbose=2)

    else:
        raise ValueError()


if __name__ == '__main__':

    main(
        num_filters=16,
        kernel_size=3,
        save_path='../dat/tmp_weights/model_2/weights',
        train_t=0.5,  # when train
        # test_ts=[0.1, 0.5, 1.0, 5.0, 10., 50., 100.],  # when test
    )
