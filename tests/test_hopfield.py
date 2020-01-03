"""It is find that, for Hopfield layer, test accuracy does converge as the
integration time goes longer."""

import logging
import numpy as np
import tensorflow as tf
from node.fix_grid import RKSolver
from node.utils.nn.hopfield import HopfieldLayer


# for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def get_compiled_model(num_filters, kernel_size, t, save_path=None):
    model = tf.keras.Sequential([
        tf.keras.layers.Input([28, 28]),
        tf.keras.layers.Reshape([28, 28, 1]),

        # Down-sampling
        tf.keras.layers.Conv2D(num_filters, kernel_size),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(),

        HopfieldLayer(num_filters, kernel_size, RKSolver(0.1), t),

        # Classification
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax'),
    ])
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    if save_path is not None:
        model.load_weights(save_path)

    return model


def main(num_filters, kernel_size, save_path,
         train_t=None, test_ts=None):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if train_t:
        model = get_compiled_model(
            num_filters, kernel_size, train_t, save_path)
        model.summary()
        model.fit(x_train, y_train, epochs=5,
                  validation_data=(x_test, y_test))
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
        num_filters=32,
        kernel_size=3,
        save_path='../dat/tmp_weights/model_2',
        train_t=None,  # when test
        # train_t=0.2,  # when train
        test_ts=[0.1, 0.5, 1.0, 5.0, 10., 50., 100.],
    )
