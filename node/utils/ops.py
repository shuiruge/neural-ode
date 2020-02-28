import tensorflow as tf


def swapaxes(x, axis1, axis2):
  rank = len(tf.shape(x))
  if axis1 < 0:
    axis1 = rank + axis1
  if axis2 < 0:
    axis2 = rank + axis2

  perm = []
  for axis in range(rank):
    if axis == axis1:
      perm.append(axis2)
    elif axis == axis2:
      perm.append(axis1)
    else:
      perm.append(axis)

  return tf.transpose(x, perm)
