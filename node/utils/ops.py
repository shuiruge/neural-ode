import tensorflow as tf


def swapaxes(a, axis1, axis2):
    shape = a.get_shape().as_list()
    perm = []
    for axis, dim in enumerate(shape):
        if axis == axis1:
            perm.append(axis2)
        elif axis == axis2:
            perm.append(axis1)
        else:
            perm.append(axis)
    return tf.transpose(a, perm)
