"""Definitions"""

import tensorflow as tf
from typing import Union, List, Callable

Time = float
PhasePoint = Union[tf.Tensor, List[tf.Tensor]]
PhaseVectorField = Callable[[PhasePoint, Time], PhasePoint]


class ODESolver:
    r"""
    Definition
    ----------
    $\text{ode_solver}(f, t_0, t_N, z(t_0)) := z(t_0) + \int_{t_0}^{t_N} f(z(t), t) dt$  # noqa:E501
    which is exectly the $z(t_N)$.
    """

    def __call__(self, phase_vector_field):
        """Returns a function that pushes the initial phase point to the final
        along the phase vector field.

        Why So Strange:
            This somehow strange signature is for TF's efficiency.
            For TF>=2, it compiles python code to graph just in time,
            demanding that all the arguments and outputs are `tf.Tensor`s
            or lists of `tf.Tensor`s, and no function.

        Args:
            phase_vector_field: PhaseVectorField

        Returns: Callable[[Time, Time, PhasePoint], PhasePoint]
            Args:
                start_time: Time
                end_time: Time
                initial_phase_point: PhasePoint
            Returns: PhasePoint
        """
        return NotImplemented
