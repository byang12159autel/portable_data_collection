"""Runnable entry-point scripts that compose the three components.

Single-component CLIs (e.g. ``python -m calibration.fisheye``,
``python -m gripper.pipeline``) stay inside their owning component.
This folder is for runners that either span more than one component
(``rig_replay.py`` runs ``pose_6d`` + ``gripper`` together) or are
general-purpose preview tools that aren't owned by any one component.
"""
