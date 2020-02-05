import tensorflow as tf


def nest_map(fn):  # TODO: add example.
  r"""Decorator converting `fn` to a nest map."""

  def get_name(kwargs):
    name = kwargs.get('name', None)
    if not name:
      # for `tf.function` decorated
      name = fn.__dict__.get('_name', None)
    if not name:
      name = fn.__name__
    return name

  def nest_fn(*args, **kwargs):
    r"""All args shall share the same nesting structure.

    **ONLY SUPPORTS LIST NESTING.**
    """
    name = get_name(kwargs)

    def kwargs_filled_fn(*args):
      return fn(*args, **kwargs)

    with tf.name_scope(name):
      return tf.nest.map_structure(kwargs_filled_fn, *args)

  return nest_fn
