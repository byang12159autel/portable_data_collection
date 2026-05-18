"""Marker / plane / axes overlays drawn through a homography.

Every preview app that has an ArUco pose draws (a) the detected quad,
(b) the marker's XYZ axes, and (c) a metric grid on the marker plane.
These helpers do that, given a ``H_plane_to_img`` from
``core.geometry.homography_from_aruco_pose``.
"""

from __future__ import annotations

import cv2
import numpy as np

from core.geometry import apply_homography, apply_homography_batch


def draw_detection(img: np.ndarray, corners: np.ndarray, tag_id: int) -> None:
    """Outline the marker quad and label its corners + ID."""
    pts = corners.astype(np.int32)
    cv2.polylines(img, [pts], True, (0, 255, 0), 2)
    for label, (x, y) in zip(("TL", "TR", "BR", "BL"), pts):
        cv2.circle(img, (int(x), int(y)), 4, (0, 255, 255), -1)
        cv2.putText(img, label, (int(x) + 6, int(y) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    cx, cy = pts.mean(axis=0).astype(int)
    cv2.putText(img, f"id={tag_id}", (cx - 20, cy + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)


def draw_axes_via_homography(img: np.ndarray, H_plane_to_img: np.ndarray,
                             length_m: float) -> None:
    """Draw the marker's XY axes by warping plane points through H.

    Z is handled separately by ``draw_z_axis`` (it needs the full 3D
    projection, since it's out of plane).
    """
    origin = apply_homography(H_plane_to_img, (0.0, 0.0))
    x_end = apply_homography(H_plane_to_img, (length_m, 0.0))
    y_end = apply_homography(H_plane_to_img, (0.0, length_m))
    o = tuple(int(v) for v in origin)
    cv2.arrowedLine(img, o, tuple(int(v) for v in x_end), (0, 0, 255), 3,
                    tipLength=0.15)  # X red
    cv2.arrowedLine(img, o, tuple(int(v) for v in y_end), (0, 255, 0), 3,
                    tipLength=0.15)  # Y green


def draw_z_axis(img: np.ndarray, K: np.ndarray, rvec: np.ndarray,
                tvec: np.ndarray, length_m: float) -> None:
    """Z axis needs the full 3D projection; draw on top of the homography axes."""
    pts_3d = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, length_m]], dtype=np.float32
    )
    pts_2d, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, np.zeros(5))
    p0 = tuple(int(v) for v in pts_2d[0, 0])
    p1 = tuple(int(v) for v in pts_2d[1, 0])
    cv2.arrowedLine(img, p0, p1, (255, 0, 0), 3, tipLength=0.15)  # Z blue


def draw_plane_grid(img: np.ndarray, H_plane_to_img: np.ndarray,
                    extent_m: float, step_m: float) -> None:
    """Draw a square grid on the marker plane via the homography."""
    h, w = img.shape[:2]
    ticks = np.arange(-extent_m, extent_m + step_m * 0.5, step_m)
    for y in ticks:
        pts = apply_homography_batch(
            H_plane_to_img, np.column_stack([ticks, np.full_like(ticks, y)])
        ).astype(np.int32)
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            if 0 <= x0 < w and 0 <= y0 < h and 0 <= x1 < w and 0 <= y1 < h:
                cv2.line(img, (x0, y0), (x1, y1), (255, 200, 80), 1, cv2.LINE_AA)
    for x in ticks:
        pts = apply_homography_batch(
            H_plane_to_img, np.column_stack([np.full_like(ticks, x), ticks])
        ).astype(np.int32)
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            if 0 <= x0 < w and 0 <= y0 < h and 0 <= x1 < w and 0 <= y1 < h:
                cv2.line(img, (x0, y0), (x1, y1), (255, 200, 80), 1, cv2.LINE_AA)
