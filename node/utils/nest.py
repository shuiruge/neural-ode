import tensorflow as tf


def nest_map(fn):  # TODO: add example.
  r"""Decorator converting `fn` to a nest map."""

  def nest_fn(*args, **kwargs):
    """All args shall share the same nesting structure."""

    def kwargs_filled_fn(*args):
      return fn(*args, **kwargs)

    return tf.nest.map_structure(kwargs_filled_fn, *args)

  return nest_fn
