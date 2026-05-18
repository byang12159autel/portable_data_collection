"""``LearnedLayoutEstimator`` -- ``PoseEstimator`` implementation.

Per frame: from the shared ``Detections``, pick every marker that's in
the learned layout, stack each marker's 3D world corners (its local
object points transformed by ``T_world_marker``) against its image
corners, and run one pooled ``cv2.solvePnP``. The result inverts to
``T_world_camera``.

Robust to the anchor marker falling out of view -- only requires that
*some* marker in the layout is visible.
"""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np

from core.markers import marker_object_points
from core.pipeline import Detections, FramePose
from pose_6d.layout import LearnedLayout, rvec_tvec_to_T


@dataclasses.dataclass
class LearnedLayoutEstimator:
    """Pooled multi-marker PnP against a pre-learned world layout.

    ``dist_coeffs`` defaults to zero — i.e. the upstream rectifier is
    expected to have produced a clean pinhole frame so PnP can ignore
    distortion. For pipelines whose rectifier leaves residual
    distortion (e.g. the single-stage path in
    ``pose_6d/learned_layout.py``'s CLI), pass the calibrated pinhole
    ``D`` here.
    """

    layout: LearnedLayout
    dist_coeffs: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(5, dtype=np.float64)
    )

    def __call__(self, detections: Detections, K: np.ndarray) -> FramePose:
        if detections.ids is None or len(detections.ids) == 0:
            return FramePose(T_world_camera=None, n_inliers=0)

        ids_flat = np.asarray(detections.ids).flatten()
        obj_world_list: list[np.ndarray] = []
        img_list: list[np.ndarray] = []
        for i, tid in enumerate(ids_flat):
            tid = int(tid)
            if tid not in self.layout.T_world_marker:
                continue
            cfg = self.layout.marker_configs[tid]
            obj_local = marker_object_points(cfg.size)  # (4, 3)
            obj_h = np.hstack([obj_local, np.ones((4, 1), dtype=np.float64)])
            obj_world = (self.layout.T_world_marker[tid] @ obj_h.T).T[:, :3].astype(np.float32)
            img_pts = np.asarray(detections.corners[i]).reshape(4, 2).astype(np.float32)
            obj_world_list.append(obj_world)
            img_list.append(img_pts)

        if not obj_world_list:
            return FramePose(T_world_camera=None, n_inliers=0)

        obj_pts = np.vstack(obj_world_list)
        img_pts = np.vstack(img_list)
        flags = (
            cv2.SOLVEPNP_IPPE_SQUARE if len(obj_pts) == 4
            else cv2.SOLVEPNP_ITERATIVE
        )
        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, self.dist_coeffs, flags=flags)
        if not ok:
            return FramePose(T_world_camera=None, n_inliers=0)

        T_cam_world = rvec_tvec_to_T(rvec, tvec)
        return FramePose(
            T_world_camera=np.linalg.inv(T_cam_world),
            n_inliers=len(obj_world_list),
        )
