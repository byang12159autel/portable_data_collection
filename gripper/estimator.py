"""``HingeAngleEstimator`` -- ``GripperEstimator`` implementation.

Per frame: pick the anchor marker out of the shared ``Detections``, run
single-marker PnP for the marker-plane pose, lift dot detections into
plane coordinates, and compute the in-plane chopstick hinge angle.

This is the library facade for the gripper component. The CLI runner
``gripper/pipeline.py`` does the same per-frame work inline but adds
drawing + mp4/viser output; the composed ``RigPipeline`` in
``runners/rig_replay.py`` uses this class directly so pose + gripper
share one frame stream.
"""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np

from core.geometry import apply_homography_batch, homography_from_aruco_pose
from core.markers import marker_object_points
from core.pipeline import Detections, GripperState
from gripper.hinge import (
    hinge_angle_in_plane,
    run_detection_pass,
    select_four_dots,
)


@dataclasses.dataclass
class HingeAngleEstimator:
    """``GripperEstimator`` for the chopstick hinge angle.

    Identify the anchor marker by ``marker_id`` (must be one of the IDs
    the upstream detector is configured to find). The 4 chopstick
    reference dots sit slightly behind the marker face -- pass that
    offset in metres as ``dot_plane_z_offset_m`` (negative goes into the
    marker face) so the homography lifts onto the correct plane.

    Dot thresholds are interpreted on the warped grid when
    ``detect_in_plane=True`` (the default), so they should match
    ``gripper/tune.py --detect-in-plane`` values.
    """

    marker_id: int
    marker_size: float
    dot_plane_z_offset_m: float = -0.005
    detect_in_plane: bool = True
    detect_plane_px: int = 800
    detect_plane_extent_factor: float = 4.0
    white_dot_threshold: int = 175
    white_dot_min_area: int = 50
    white_dot_max_area: int = 700
    white_dot_min_circularity: float = 0.55
    black_dot_threshold: int = 80
    black_dot_min_area: int = 70
    black_dot_max_area: int = 700
    black_dot_min_circularity: float = 0.55

    def __call__(
        self,
        rectified: np.ndarray,
        detections: Detections,
        K: np.ndarray,
    ) -> GripperState:
        corners = self._select_anchor_corners(detections)
        if corners is None:
            return GripperState()

        ok_pnp, rvec, tvec = cv2.solvePnP(
            marker_object_points(self.marker_size), corners,
            K, np.zeros(5, dtype=np.float64),
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok_pnp:
            return GripperState()

        H_plane_to_img, H_img_to_plane = homography_from_aruco_pose(
            K, rvec, tvec, z_offset_m=self.dot_plane_z_offset_m,
        )
        plane_extent = self.marker_size * self.detect_plane_extent_factor

        white_dets, white_plane = run_detection_pass(
            rectified, corners, H_plane_to_img, H_img_to_plane,
            polarity="light",
            detect_in_plane=self.detect_in_plane,
            extent_m=plane_extent,
            px=self.detect_plane_px,
            marker_size=self.marker_size,
            threshold=self.white_dot_threshold,
            min_area=self.white_dot_min_area,
            max_area=self.white_dot_max_area,
            min_circularity=self.white_dot_min_circularity,
        )
        black_dets, _ = run_detection_pass(
            rectified, corners, H_plane_to_img, H_img_to_plane,
            polarity="dark",
            detect_in_plane=self.detect_in_plane,
            extent_m=plane_extent,
            px=self.detect_plane_px,
            marker_size=self.marker_size,
            threshold=self.black_dot_threshold,
            min_area=self.black_dot_min_area,
            max_area=self.black_dot_max_area,
            min_circularity=self.black_dot_min_circularity,
        )

        state = GripperState(
            H_plane_to_img=H_plane_to_img,
            H_img_to_plane=H_img_to_plane,
            n_white_dots=len(white_dets),
            n_black_dots=len(black_dets),
        )
        if len(white_dets) < 4:
            return state

        if white_plane is None:
            centers_img = np.array([d["center"] for d in white_dets])
            white_plane = apply_homography_batch(H_img_to_plane, centers_img)
        _, centers_plane = select_four_dots(white_dets, white_plane)

        try:
            angle_deg, ordered = hinge_angle_in_plane(centers_plane)
        except ValueError:
            return state

        state.centers_plane = centers_plane
        state.ordered_centers = ordered
        state.angle_deg = angle_deg
        return state

    def _select_anchor_corners(
        self, detections: Detections,
    ) -> np.ndarray | None:
        """Return the 4 image corners of ``self.marker_id`` from detections, or None."""
        if detections.ids is None or len(detections.ids) == 0:
            return None
        idx = np.where(np.asarray(detections.ids).flatten() == self.marker_id)[0]
        if len(idx) == 0:
            return None
        return np.asarray(detections.corners[int(idx[0])]).reshape(4, 2).astype(np.float32)
