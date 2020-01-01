import tensorflow as tf


@tf.function
def accuracy(y_true, y_pred):
    output_dtype = y_pred.dtype
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)
    return tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), output_dtype))


@tf.function
def process_datum(X, y=None):
    X = tf.cast(X, tf.float32)
    X = X / 255.
    X = tf.reshape(X, [28 * 28])
    if y is None:
        return X
    else:
        y = tf.one_hot(y, 10)
        y = tf.cast(y, tf.float32)
        return X, y


def get_dataset():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return x_train, y_train, x_test, y_test
