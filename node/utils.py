import tensorflow as tf


def _is_nested(x):
    """Auxillary function of `nest_map`."""
    return isinstance(x, (list, tuple))


def _check_same_structure(*args):
    """Auxillary function of `nest_map`."""
    if len(args) == 1:
        return

    first, *rests = args

    if not _is_nested(first):
        for arg in rests:
            if _is_nested(arg):
                raise ValueError()
        return

    for arg in rests:
        if not _is_nested(arg):
            raise ValueError()
        if len(arg) != len(first):
            raise ValueError()
    return


def nest_map(fn, *args):  # TODO: add example.
    """All args shall share the same nesting structure."""
    _check_same_structure(*args)

    if not _is_nested(args[0]):
        return fn(*args)

    return [nest_map(fn, *subargs) for subargs in zip(*args)]


def tracer(solver, fn):
    """
    Args:
        solver: ODESolver
        fn: Callable[[Time, tf.Tensor], tf.Tensor]

    Returns: Callable[[Time, Time, Time, tf.Tensor], tf.TensorArray]
        The arguments are start time, end time, time difference, and
        initial phase point. Returns the trajectory.
    """
    forward = solver(fn)

    @tf.function
    def trace(t0, t1, dt, x):
        dt = tf.where(t1 > t0, dt, -dt)
        num_grids = int((t1 - t0) / dt + 1)
        ts = tf.linspace(t0, t1, num_grids)

        i = 0
        xs = tf.TensorArray(x.dtype, size=num_grids)
        xs = xs.write(i, x)

        ts = tf.linspace(t0, t1, num_grids)
        for t in ts[:-1]:
            x = forward(t, t + dt, x)
            i += 1
            xs = xs.write(i, x)
        return xs.stack()

    return trace
