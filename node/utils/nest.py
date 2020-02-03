import tensorflow as tf


def nest_map(fn):  # TODO: add example.
    r"""Decorator converting `fn` to a nest map."""

    def nest_fn(*args):
        r"""All args shall share the same nesting structure.

        **ONLY SUPPORTS LIST NESTING.**
        """
        return _nest_map_recur(fn, *args)

    return nest_fn


def _nest_map_recur(fn, *args):
    r"""All args shall share the same nesting structure.

    **ONLY SUPPORTS LIST NESTING.**
    """
    _check_same_structure(*args)

    if not isinstance(args[0], list):
        return fn(*args)

    return [_nest_map_recur(fn, *subargs) for subargs in zip(*args)]


def _check_same_structure(*args):
    first_arg, *rest_args = args
    for arg in rest_args:
        tf.nest.assert_same_structure(first_arg, arg)
