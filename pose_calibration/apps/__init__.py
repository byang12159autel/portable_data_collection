"""Entry-point scripts -- viser viewers + ROS capture node.

Each module composes a pipeline by importing concrete implementations
from :mod:`pose_calibration.markers`, :mod:`pose_calibration.calibration`,
and :mod:`pose_calibration.pose`. Swap a stage by editing one constructor
call.
"""
