"""``LearnedLayout`` + supporting SE(3) math.

The runtime pose estimator (:mod:`pose_6d.estimator`) needs a static
``T_world_marker`` dict to do pooled PnP. The recipe for building that
dict from a video — single-tag PnP per frame, then quaternion-mean
across all frames where the anchor + each non-anchor marker are
co-visible — lives here.

``LearnedLayout.from_observations`` consumes pre-computed per-frame
``T_camera_marker`` observations (the kind ``detect_per_marker_pnp``
returns), so callers iterate the video once and choose what else to
keep alongside (overlay corners, frame indices, etc.).
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path  # noqa: TC003 — runtime use in save/load signatures

import cv2
import numpy as np
import yaml

from core.markers import MarkerConfig, detect_aruco_markers, marker_object_points


# ---------------------------------------------------------------------------
# SE(3) helpers
# ---------------------------------------------------------------------------


def rvec_tvec_to_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Compose a 4x4 homogeneous transform from solvePnP outputs."""
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec.ravel()
    return T


def R_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> (w, x, y, z) unit quaternion."""
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * math.sqrt(1.0 + tr)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float64)


def quat_to_R(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
         [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
         [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)]],
        dtype=np.float64,
    )


def T_to_wxyz_xyz(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a 4x4 transform into (wxyz quaternion, xyz translation)."""
    return R_to_quat(T[:3, :3]), T[:3, 3].astype(np.float64)


def avg_T(Ts: list[np.ndarray]) -> np.ndarray:
    """Mean of 4x4 transforms: arithmetic mean translation, Markley 2007
    quaternion mean for rotation (dominant eigenvector of sum(q qT))."""
    if len(Ts) == 1:
        return Ts[0].copy()
    ts = np.stack([T[:3, 3] for T in Ts])
    qs = np.stack([R_to_quat(T[:3, :3]) for T in Ts])
    for i in range(1, len(qs)):
        if np.dot(qs[0], qs[i]) < 0:
            qs[i] = -qs[i]
    M = qs.T @ qs
    _, V = np.linalg.eigh(M)
    q_avg = V[:, -1]
    T_avg = np.eye(4, dtype=np.float64)
    T_avg[:3, :3] = quat_to_R(q_avg)
    T_avg[:3, 3] = ts.mean(axis=0)
    return T_avg


# ---------------------------------------------------------------------------
# Per-frame detection + per-marker PnP
# ---------------------------------------------------------------------------


def detect_per_marker_pnp(
    img: np.ndarray,
    marker_configs: dict[int, MarkerConfig],
    K: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], list[np.ndarray], list[int]]:
    """Detect every configured marker and run single-tag PnP per marker.

    Returns ``(obs, all_corners, all_ids)`` where:

      - ``obs`` is ``{id: (T_camera_marker (4x4), img_pts (4, 2) float32)}``
        for markers whose PnP succeeded.
      - ``all_corners`` / ``all_ids`` carry every detected marker (even ones
        where PnP failed) so the caller can draw a complete detection overlay.
    """
    by_dict: dict[int, list[int]] = {}
    for cfg in marker_configs.values():
        by_dict.setdefault(cfg.cv2_dictionary, []).append(cfg.id)

    obs: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    all_corners: list[np.ndarray] = []
    all_ids: list[int] = []
    for dict_id, ids in by_dict.items():
        corners, det_ids = detect_aruco_markers(
            img, marker_dict=dict_id, allowed_ids=set(ids),
        )
        if det_ids is None:
            continue
        for c, tid in zip(corners, det_ids.flatten()):
            tid = int(tid)
            cfg = marker_configs[tid]
            img_pts = c.reshape(4, 2).astype(np.float32)
            obj = marker_object_points(cfg.size)
            ok, rvec, tvec = cv2.solvePnP(
                obj, img_pts, K, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if ok:
                obs[tid] = (rvec_tvec_to_T(rvec, tvec), img_pts)
            all_corners.append(c)
            all_ids.append(tid)
    return obs, all_corners, all_ids


# ---------------------------------------------------------------------------
# LearnedLayout
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class LearnedLayout:
    """A static world layout learned from co-visible single-tag PnP samples.

    The anchor marker defines the world origin (``T_world_anchor = I``).
    For every other marker the layout averages ``inv(T_cam_anchor) @
    T_cam_marker`` across every frame where the anchor and that marker
    were co-visible.

    Markers that are never co-visible with the anchor are omitted from
    ``T_world_marker`` — the pose estimator simply ignores them.
    """

    T_world_marker: dict[int, np.ndarray]
    marker_configs: dict[int, MarkerConfig]
    anchor_id: int

    @classmethod
    def from_observations(
        cls,
        observations: list[dict[int, tuple[np.ndarray, np.ndarray]]],
        marker_configs: dict[int, MarkerConfig],
        anchor_id: int = -1,
    ) -> "LearnedLayout":
        """Build a layout from per-frame per-marker PnP observations.

        ``observations[i]`` is ``{id: (T_camera_marker, img_pts)}`` for
        frame ``i`` (see :func:`detect_per_marker_pnp`). Frames with no
        detected markers contribute an empty dict.

        ``anchor_id=-1`` picks the lowest-id marker that's ever observed.
        Raises ``SystemExit`` if no configured marker was seen, or if the
        requested ``anchor_id`` never appears.
        """
        seen_ids = sorted({tid for obs in observations for tid in obs})
        if not seen_ids:
            raise SystemExit("no configured markers detected in any frame")
        if anchor_id < 0:
            anchor_id = seen_ids[0]
        elif anchor_id not in seen_ids:
            raise SystemExit(
                f"anchor id {anchor_id} never detected; seen: {seen_ids}"
            )

        samples: dict[int, list[np.ndarray]] = {tid: [] for tid in seen_ids}
        samples[anchor_id].append(np.eye(4, dtype=np.float64))
        for obs in observations:
            if anchor_id not in obs:
                continue
            T_anchor_cam = np.linalg.inv(obs[anchor_id][0])
            for tid, (T_cam_m, _) in obs.items():
                if tid == anchor_id:
                    continue
                samples[tid].append(T_anchor_cam @ T_cam_m)
        T_world_marker: dict[int, np.ndarray] = {}
        for tid, sams in samples.items():
            if not sams:
                continue
            T_world_marker[tid] = avg_T(sams)
        return cls(
            T_world_marker=T_world_marker,
            marker_configs=marker_configs,
            anchor_id=anchor_id,
        )

    def save(self, path: Path) -> None:
        """Write the layout to a self-contained yaml.

        Format mirrors the inputs the layout was learned from (one entry
        per marker carrying ``id`` / ``size`` / ``dictionary``) plus a
        ``T_world_marker`` block on each entry (translation in metres +
        wxyz unit quaternion) and a top-level ``anchor_id``. Re-readable
        with :meth:`load` without needing the original
        ``--world-marker-configs`` YAMLs.
        """
        data: dict = {
            "anchor_id": int(self.anchor_id),
            "markers": {},
        }
        for tid in sorted(self.T_world_marker):
            cfg = self.marker_configs[tid]
            wxyz, xyz = T_to_wxyz_xyz(self.T_world_marker[tid])
            data["markers"][f"tag{tid}"] = {
                "id": int(cfg.id),
                "size": float(cfg.size),
                "dictionary": cfg.dictionary,
                "label": cfg.label,
                "T_world_marker": {
                    "translation": [float(v) for v in xyz],
                    "quat_wxyz": [float(v) for v in wxyz],
                },
            }
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)

    @classmethod
    def load(cls, path: Path) -> "LearnedLayout":
        """Inverse of :meth:`save`."""
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        anchor_id = int(data["anchor_id"])
        T_world_marker: dict[int, np.ndarray] = {}
        marker_configs: dict[int, MarkerConfig] = {}
        known = {f.name for f in dataclasses.fields(MarkerConfig)}
        for entry in data.get("markers", {}).values():
            tid = int(entry["id"])
            cfg_fields = {k: v for k, v in entry.items() if k in known}
            marker_configs[tid] = MarkerConfig(**cfg_fields)
            T = np.eye(4, dtype=np.float64)
            block = entry["T_world_marker"]
            q = np.asarray(block["quat_wxyz"], dtype=np.float64)
            T[:3, :3] = quat_to_R(q)
            T[:3, 3] = np.asarray(block["translation"], dtype=np.float64)
            T_world_marker[tid] = T
        return cls(
            T_world_marker=T_world_marker,
            marker_configs=marker_configs,
            anchor_id=anchor_id,
        )
