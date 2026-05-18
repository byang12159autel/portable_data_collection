"""Pure hinge-angle math + plane-space dot detection (no I/O, no drawing).

``hinge_angle_in_plane`` is the geometric heart of the gripper-state
estimator: given four dot positions in marker-plane metres, return the
angle between the two chopstick segments. The supporting functions
``detect_dots_in_plane``, ``run_detection_pass``, ``select_four_dots``
do the plane-space warp + filter + 4-dot selection that produces those
positions from a single rectified frame and the marker homographies.

Drawing (chopstick segment overlays, bird's-eye composition) is
deliberately not here — see ``gripper/pipeline.py`` for those.
"""

from __future__ import annotations

import cv2
import numpy as np

from core.geometry import apply_homography
from gripper.dots import (
    detect_black_circular_dots,
    detect_white_circular_dots,
)


def hinge_angle_in_plane(
    centers_plane: np.ndarray,
) -> tuple[float, dict[str, np.ndarray]]:
    """Hinge angle between two chopstick segments, expressed in plane coords.

    ``centers_plane`` is an ``(4, 2)`` array of dot positions in the marker
    plane (metres, +X right, +Y up). The four dots are split into a left
    pair and a right pair by the median X coordinate (robust to the marker
    not sitting exactly at the chopstick axis of symmetry), then each pair
    is ordered top->bottom by plane Y.

    Returns ``(angle_deg, ordered)`` where ``ordered`` carries the four
    points keyed ``left_top``, ``left_bottom``, ``right_top``,
    ``right_bottom`` (each ``(2,)`` plane-coord arrays).
    """
    if len(centers_plane) != 4:
        raise ValueError(f"expected 4 dots, got {len(centers_plane)}")

    med_x = float(np.median(centers_plane[:, 0]))
    left = centers_plane[centers_plane[:, 0] < med_x]
    right = centers_plane[centers_plane[:, 0] >= med_x]
    if len(left) != 2 or len(right) != 2:
        raise ValueError(f"split failed: {len(left)} left, {len(right)} right")

    # +Y is "up" in plane coords; sort ascending so [0]=bottom, [1]=top.
    left_bottom, left_top = left[np.argsort(left[:, 1])]
    right_bottom, right_top = right[np.argsort(right[:, 1])]

    a = left_top - left_bottom
    b = right_top - right_bottom
    cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta))), {
        "left_top": left_top,
        "left_bottom": left_bottom,
        "right_top": right_top,
        "right_bottom": right_bottom,
    }


def filter_dots_outside_marker(
    detections: list[dict], marker_quad: np.ndarray,
) -> list[dict]:
    """Drop detections whose centre falls inside the marker quad."""
    quad = marker_quad.astype(np.float32)
    out = []
    for det in detections:
        cx, cy = det["center"]
        if cv2.pointPolygonTest(quad, (float(cx), float(cy)), False) < 0:
            out.append(det)
    return out


def detect_dots_in_plane(
    undistorted: np.ndarray,
    H_plane_to_img: np.ndarray,
    H_img_to_plane: np.ndarray,
    *,
    polarity: str,
    extent_m: float,
    px: int,
    marker_size: float,
    threshold: int,
    min_area: int,
    max_area: int,
    min_circularity: float,
) -> tuple[list[dict], np.ndarray]:
    """Warp the frame to a fronto-parallel plane view, run dot detection there.

    In the warped grid each dot is (to first order) a circle again — so
    the circularity filter is no longer biased by viewing tilt and the
    detected centre is the true geometric centre of the dot, not the
    centre of its projected ellipse.

    Returns ``(detections, centers_plane)`` where each detection's
    ``"center"`` is rewritten as its image-space pixel coordinate (via
    ``H_plane_to_img``) so downstream drawing on the main canvas Just
    Works. The metric plane coordinate is stashed under
    ``"plane_center"`` for inspection.

    ``min_area`` / ``max_area`` are interpreted on the **warped** grid,
    not the source frame — use ``gripper/tune.py --detect-in-plane`` to
    pick values that match this resolution.
    """
    s = px / (2.0 * extent_m)
    H_plane_to_warped = np.array(
        [[s, 0.0, px / 2.0],
         [0.0, -s, px / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    H_img_to_warped = H_plane_to_warped @ H_img_to_plane
    warped = cv2.warpPerspective(
        undistorted, H_img_to_warped, (px, px),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
    )
    detector = (
        detect_white_circular_dots if polarity == "light"
        else detect_black_circular_dots
    )
    dets, _ = detector(
        warped, threshold=threshold, min_area=min_area,
        max_area=max_area, min_circularity=min_circularity,
    )
    if not dets:
        return [], np.empty((0, 2))

    # Warped pixel -> plane metres. Y was flipped during the warp.
    centers_warped = np.array([d["center"] for d in dets])
    plane = np.column_stack([
        (centers_warped[:, 0] - px / 2.0) / s,
        -(centers_warped[:, 1] - px / 2.0) / s,
    ])
    # Drop dots inside the marker square (origin-centered).
    half = marker_size / 2.0
    keep = (np.abs(plane[:, 0]) > half) | (np.abs(plane[:, 1]) > half)

    dets_out: list[dict] = []
    plane_out: list[np.ndarray] = []
    for det, p, k in zip(dets, plane, keep):
        if not k:
            continue
        img_uv = apply_homography(H_plane_to_img, p)
        det = dict(det)  # shallow copy so caller's input stays intact
        det["center"] = np.asarray(img_uv, dtype=float)
        det["plane_center"] = p
        dets_out.append(det)
        plane_out.append(p)
    return dets_out, (
        np.asarray(plane_out) if plane_out else np.empty((0, 2))
    )


def run_detection_pass(
    undistorted: np.ndarray,
    marker_corners: np.ndarray | None,
    H_plane_to_img: np.ndarray | None,
    H_img_to_plane: np.ndarray | None,
    *,
    polarity: str,
    detect_in_plane: bool,
    extent_m: float,
    px: int,
    marker_size: float,
    threshold: int,
    min_area: int,
    max_area: int,
    min_circularity: float,
) -> tuple[list[dict], np.ndarray | None]:
    """One dot-detection pass of a single polarity.

    Returns ``(detections, centers_plane)``. ``centers_plane`` is set
    when plane-mode warps were used (the warp pass also yields plane
    coords for free); ``None`` for the image-space path, where the
    caller projects through ``H_img_to_plane`` later.
    """
    if detect_in_plane and H_plane_to_img is not None and H_img_to_plane is not None:
        return detect_dots_in_plane(
            undistorted, H_plane_to_img, H_img_to_plane,
            polarity=polarity,
            extent_m=extent_m, px=px, marker_size=marker_size,
            threshold=threshold, min_area=min_area,
            max_area=max_area, min_circularity=min_circularity,
        )
    detector = (
        detect_white_circular_dots if polarity == "light"
        else detect_black_circular_dots
    )
    dets, _ = detector(
        undistorted, threshold=threshold,
        min_area=min_area, max_area=max_area,
        min_circularity=min_circularity,
    )
    if marker_corners is not None:
        dets = filter_dots_outside_marker(dets, marker_corners)
    return dets, None


def select_four_dots(
    detections: list[dict], centers_plane: np.ndarray,
) -> tuple[list[dict], np.ndarray]:
    """If more than 4 dots survived, keep the four closest to the marker."""
    if len(detections) <= 4:
        return detections, centers_plane
    dists = np.linalg.norm(centers_plane, axis=1)
    keep = np.argsort(dists)[:4]
    return [detections[i] for i in keep], centers_plane[keep]
