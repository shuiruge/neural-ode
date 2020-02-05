import tensorflow as tf


def nest_map(fn):  # TODO: add example.
  r"""Decorator converting `fn` to a nest map."""

  def nest_fn(*args, **kwargs):
    r"""All args shall share the same nesting structure.

    **ONLY SUPPORTS LIST NESTING.**
    """

    def kwargs_filled_fn(*args):
      return fn(*args, **kwargs)

    return tf.nest.map_structure(kwargs_filled_fn, *args)

  return nest_fn
