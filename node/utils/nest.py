import tensorflow as tf


def nest_map(fn, *args):  # TODO: add example.
    r"""All args shall share the same nesting structure.

    **ONLY SUPPORTS LIST NESTING.**
    """
    _check_same_structure(*args)

    if not isinstance(args[0], list):
        return fn(*args)

    return [nest_map(fn, *subargs) for subargs in zip(*args)]


def _check_same_structure(*args):
    first_arg, *rest_args = args
    for arg in rest_args:
        tf.nest.assert_same_structure(first_arg, arg)
