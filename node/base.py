"""Definitions"""

import tensorflow as tf


class Time:
  """Type that represents float scalar."""


class PhasePoint:
  """Type that represents `tf.Tensor` or nest structure of `tf.Tensor`."""


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


class Diagnostics:
  """Diagnostics of the solving process of an ODE solver.

  ODE solver specific.
  """


class Results:
    """Results of ODE solver.

    Parameters
    ----------
    time : Time
    phase_point : PhasePoint
    diagnostics : Diagnostics
    """

  def __init__(self, time, phase_point, diagnostics):
    self.time = time
    self.phase_point = phase_point
    self.diagnostics = diagnostics


class ODESolver:

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
      Results
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
      Results
      """
      return NotImplemented

    return forward
