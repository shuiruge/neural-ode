import tensorflow as tf


@tf.function(input_signature=[
    tf.TensorSpec(shape=[None, 3], dtype=tf.float32),
    tf.TensorSpec(shape=[None, 3], dtype=tf.float32),
])
def f(x, y):
    print(f'tracing f: {x.shape}, {y.shape}')

    @tf.function
    def g(x):
        print(f'tracing g: {x.shape}')
        return x + y

    return g(x + 1)


x = tf.random.uniform(shape=[2, 3])
y = tf.random.uniform(shape=[2, 3])
f(x, y)

x = tf.random.uniform(shape=[3, 3])
y = tf.random.uniform(shape=[3, 3])
f(x, y)

x = tf.random.uniform(shape=[4, 3])
y = tf.random.uniform(shape=[4, 3])
f(x, y)
