import tensorflow as tf
from node.utils.nest import nest_map


def rmsprop(dynamics, gamma=1e-0, eps=1e-7):
  r"""Continuum of RMSprop.

  ```math
  Changes the original dynamics $dz^a / dt = f^a(t, z)$ to

  \begin{align}
    \frac{dz^a}{dt} & = \frac{f^a(t, z)}{\sqrt{s^a + \epsilon}}; \\
    \frac{ds^a}{dt} & = \gamma \left[ - s^a + (f^a)^2(t, z) \right],
  \end{align}

  where $s$ is the MS (mean square) vector.
  ```

  References:
    1. https://ruder.io/optimizing-gradient-descent/index.html#rmsprop
  
  Args:
    dynamics: PhaseVectorField
    gamma: float
    eps: float
  
  Returns: PhaseVectorField
  """

  @nest_map
  def ms_dynamics(ms, dz):
    return gamma * (- ms + dz ** 2)

  @nest_map
  def inv_rms(ms):
    return 1 / tf.sqrt(ms + eps)
  
  @tf.function
  def rmsprop_dynamics(t, x):
    z, ms = x
    dz = dynamics(t, z)
    dms = ms_dynamics(ms, dz)
    dz *= inv_rms(ms)
    return dz, dms

  return rmsprop_dynamics
