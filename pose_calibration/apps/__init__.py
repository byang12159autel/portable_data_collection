"""Entry-point scripts -- viser viewers + ROS capture node.

Each module composes a pipeline by importing concrete implementations
from :mod:`core.markers`, :mod:`core.rectify`, and
:mod:`pose_6d`. Swap a stage by editing one constructor
call.
"""
