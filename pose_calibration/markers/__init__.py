"""Marker detection + YAML config loaders.

Implements the ``Detector`` protocol from :mod:`pose_calibration.pipeline`.
Supports ArUco, ChArUco, AprilTag, and AprilGrid (calib.io / Kalibr) via
OpenCV's ``cv2.aruco`` module.
"""
