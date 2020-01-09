import tensorflow as tf


def find_root(fn,
              input_shape,
              initializer,
              optimizer,
              iterations,
              tolerance,
              verbose=1):
    """Finds the roots of the function `fn`.

    TODO: implements verbose.

    Args:
        fn: Callable[[tf.Tensor], tf.Tensor]
        input_shape: List[int]
            The first dimension is batch-dimension.
        initializer: tf.initializers.Initializer
        optimizer: tf.optimizers.Optimizer
        iterations: int
            Maximum of iterations.
        tolerance: float
        verbose: int
            The same as the argument `verbose` of `tf.keras.Model.fit`.

    Returns: Tuple[tf.Variable, tf.Tensor]
        For the roots and the loss-per-sample.
    """
    input_var = tf.Variable(initializer(input_shape))
    iterations = tf.convert_to_tensor(iterations)
    tolerance = tf.convert_to_tensor(tolerance)
    sample_axis = list(range(len(input_shape)))[1:]

    @tf.function
    def iterate():
        with tf.GradientTape() as g:
            loss_per_sample = tf.reduce_max(tf.square(fn(input_var)),
                                            axis=sample_axis)
        grad = g.gradient(loss_per_sample, input_var)
        optimizer.apply_gradients([(grad, input_var)])
        return loss_per_sample

    @tf.function
    def train():
        loss_per_sample = tf.ones(input_shape[0]) * float('inf')
        for step in tf.range(iterations):
            loss_per_sample = iterate()
            if tf.reduce_max(loss_per_sample) < tolerance:
                break
        return loss_per_sample

    loss_per_sample = train()
    return input_var, loss_per_sample


if __name__ == '__main__':

    @tf.function
    def fn(x):
        return (x - 1) * (x + 1)

    def initializer(shape):
        return 2 * (2 * tf.random.uniform(shape) - 1)

    optimizer = tf.optimizers.Adam(1e-1)
    print(find_root(fn, [20, 100], initializer, optimizer, 10000, 1e-3))
