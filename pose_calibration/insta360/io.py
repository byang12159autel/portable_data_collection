"""Splitter for the **live USB stream** dual-fisheye layout.

The Insta360 ROS driver receives the two lenses side-by-side in a
single 1920x960-ish frame (see ``insta360_ros_driver/src/main.cpp`` and
``equirectangular.py``). ``split_dual_fisheye`` is the inverse of that
packing.

**Not applicable to ``.insv`` files** — X4/X5 record each lens to its
own video stream inside the container, so the offline pipeline uses
``insta360.convert`` to demux them and operates on per-lens videos
directly. Keep this helper around for any ROS-side processing.
"""

from __future__ import annotations

import cv2
import numpy as np


def split_dual_fisheye(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a side-by-side dual-fisheye frame into (front, back) upright crops.

    Matches the rotation convention used by the ROS equirectangular node:
    source is ``[back | front]`` with each half rotated 90 degrees from
    upright, so the front half is un-rotated counter-clockwise and the
    back half clockwise.
    """
    if frame.ndim != 3:
        raise ValueError(f"expected HxWx3 frame, got shape {frame.shape}")
    midpoint = frame.shape[1] // 2
    back = cv2.rotate(frame[:, :midpoint], cv2.ROTATE_90_CLOCKWISE)
    front = cv2.rotate(frame[:, midpoint:], cv2.ROTATE_90_COUNTERCLOCKWISE)
    return front, back
