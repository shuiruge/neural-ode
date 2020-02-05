import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import animation
from node.utils.ops import swapaxes


def tracer(solver, fn):
  """
  Args:
    solver: ODESolver
    fn: PhaseVectorField

  Returns: Callable[[Time, Time, Time, tf.Tensor], tf.Tensor]
    The arguments are start time, end time, time difference, and
    initial phase point. Returns the trajectory.
  """
  forward = solver(fn)

  @tf.function
  def trace(t0, t1, dt, x, name='trace'):
    """
    Args:
      t0: Time
      t1: Time
      dt: Time
      x: tf.Tensor

    Returns: tf.Tensor
      If `x` has the shape `[batch_size] + event_shape`,
      then the returned trajectory has the shape
      `[batch_size, trajectory_length] + event_shape`.
    """
    with tf.name_scope(name):
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
      trajectory = xs.stack()  # [B, T] + E
      return swapaxes(trajectory, 0, 1)  # [T, B] + E

  return trace


def visualize_trajectory(trajectory, normalize=False):
  """Visualizes trajectory by imshow animation.

  Args:
    trajectory: numpy.array
      Shape `[num_frames, num_x_pixals, num_y_pixals]`.
    normalize: bool
      If true, the trajectory is normalized for each frame.

  Returns: animation.FuncAnimation
  """

  def normalize_fn(frame):
    if np.std(frame) == 0:
      return frame
    frame = (frame - np.min(frame)) / (np.max(frame) - np.min(frame))
    return frame

  if normalize:
    trajectory = [normalize_fn(_) for _ in trajectory]

  fig = plt.figure()
  ax = plt.axes()
  img = ax.imshow(trajectory[0], cmap='gray')

  def init():
    img.set_data([[]])
    return img,

  def animate(i):
    frame = trajectory[i]
    img.set_data(frame)
    return img,

  anim = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=len(trajectory), blit=True)
  return anim
