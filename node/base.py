"""Definitions"""

from collections import namedtuple


class Time:
  """Type that represents float scalar."""


class PhasePoint:
  """Type that represents float tensor or nest structure of float tensor."""


class PhaseVectorField:

  def __call__(self, time, phase_point):
    """
    Parameters
    ----------
    time : Time
    phase_point : PhasePoint

    Returns
    -------
    PhasePoint
    """
    return NotImplemented


ODEResult = namedtuple('ODEResult', 'time, phase_point, diagnostics')


class ODESolver:
  """
  Attributes
  ----------
  diagnostics : Diagnostics
  """

  def __call__(self, phase_vector_field):
    """
    Parameters
    ----------
    phase_vector_field : PhaseVectorField

    Returns
    -------
    callable
    """

    def forward(start_time, end_time, initial_phase_point):
      """
      Parameters
      ----------
      start_time : Time
      end_time : Time
      initial_phase_point : PhasePoint

      Returns
      -------
      ODEResult
      """
      return NotImplemented

    return forward


class StopCondition:
  """Determines whether stopping or not."""

  def __call__(start_time, initial_phase_point,
               current_time, current_phase_point):
    """
    Parameters
    ----------
    start_time : Time
    initial_phase_point : PhasePoint
    current_time : Time
    current_phase_point : PhasePoint

    Returns
    -------
    bool
    """
    return NotImplemented


class DynamicalODESolver:

  def __call__(self, phase_vector_field, stop_condition):
    """
    Parameters
    ----------
    phase_vector_field : PhaseVectorField
    stop_condition : StopCondition

    Returns
    -------
    callable
    """

    def forward(start_time, initial_phase_point):
      """
      Parameters
      ----------
      start_time : Time
      initial_phase_point : PhasePoint

      Returns
      -------
      ODEResult
      """
      return NotImplemented

    return forward
