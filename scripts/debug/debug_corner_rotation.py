"""Try all 4 corner-order rotations of the AprilGrid object points.

If our per-tag corner labels (TL, TR, BR, BL) are rotated relative to
what the OpenCV ArUco detector returns, calibration produces systematic
high RMS regardless of data. Whichever rotation here yields the lowest
RMS is the correct labelling.

Usage::

    pixi run python debug_corner_rotation.py <lens0_video.mp4>
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np

from pose_calibration.markers.detect import (
    apriltag_grid_object_points,
    detect_aruco_markers,
    load_apriltag_grid_configs,
)

if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(1)

video = Path(sys.argv[1])
config = Path("config/apriltag_board.yaml")
grids = load_apriltag_grid_configs(config)
_, grid_cfg = next(iter(grids.items()))
dict_id = grid_cfg.cv2_dictionary
allowed = set(grid_cfg.tag_ids)
grid_obj_pts = apriltag_grid_object_points(grid_cfg)

# Use a strict subset of frames: 2nd-or-later .insv (skip first that has
# bad coverage), every 30 frames, must have >=20 corners with good spread.
cap = cv2.VideoCapture(str(video))
if not cap.isOpened():
    print(f"could not open {video}")
    sys.exit(1)
n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"scanning {n_total} frames of {video} ({W}x{H})")

raw_obj: list[np.ndarray] = []
raw_img: list[np.ndarray] = []
idx = -1
while True:
    ok, bgr = cap.read()
    if not ok:
        break
    idx += 1
    if idx % 30 != 0:
        continue
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    corners, ids = detect_aruco_markers(rgb, marker_dict=dict_id, allowed_ids=allowed)
    if ids is None:
        continue
    # collect per-tag (obj 4x3, img 4x2) pairs, raw order
    obj_list: list[np.ndarray] = []
    img_list: list[np.ndarray] = []
    for c, tid in zip(corners, ids.flatten()):
        tid = int(tid)
        if tid not in grid_obj_pts:
            continue
        obj_list.append(grid_obj_pts[tid])  # (4, 3)
        img_list.append(c.reshape(4, 2))    # (4, 2)
    if not obj_list or sum(len(p) for p in obj_list) < 12:
        continue
    obj = np.vstack(obj_list).astype(np.float64).reshape(-1, 1, 3)
    img = np.vstack(img_list).astype(np.float64).reshape(-1, 1, 2)
    # spread check (your board occupies ~15% of frame at typical hand-held distance)
    pts = img.reshape(-1, 2)
    if pts[:, 0].max() - pts[:, 0].min() < 0.08 * W:
        continue
    if pts[:, 1].max() - pts[:, 1].min() < 0.08 * H:
        continue
    raw_obj.append(obj)
    raw_img.append(img)
cap.release()
print(f"collected {len(raw_obj)} high-quality views")

if len(raw_obj) < 5:
    print("not enough high-quality views; loosen the filter inside this script")
    sys.exit(1)

# Helper to rotate per-tag corner ordering by N positions (0/1/2/3).
def rotate_corners(obj_views: list[np.ndarray], shift: int) -> list[np.ndarray]:
    rotated = []
    for o in obj_views:
        # o has shape (4*M, 1, 3): M tags, 4 corners each
        m = o.shape[0] // 4
        reshaped = o.reshape(m, 4, 3)
        # cyclically shift the 4 corners within each tag
        shifted = np.roll(reshaped, shift, axis=1)
        rotated.append(shifted.reshape(-1, 1, 3))
    return rotated


def try_one(label: str, obj_views: list[np.ndarray]) -> float:
    f_init = W / math.pi
    K = np.array([[f_init, 0, W / 2.0], [0, f_init, H / 2.0], [0, 0, 1]], dtype=np.float64)
    D = np.zeros(4, dtype=np.float64)
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in obj_views]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in obj_views]
    flags = cv2.fisheye.CALIB_FIX_SKEW | cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    try:
        rms, _, _, _, _ = cv2.fisheye.calibrate(
            obj_views, raw_img, (W, H), K, D, rvecs, tvecs, flags, crit,
        )
        print(
            f"{label}: RMS={rms:.4f} px  fx={K[0,0]:.1f} fy={K[1,1]:.1f}  "
            f"cx={K[0,2]:.1f} cy={K[1,2]:.1f}  D={D.ravel().tolist()}"
        )
        return float(rms)
    except cv2.error as e:
        print(f"{label}: cv2.error: {e}")
        return float("inf")


print()
results = {}
for shift in range(4):
    results[shift] = try_one(f"rotation {shift}", rotate_corners(raw_obj, shift))

best_shift = min(results, key=lambda s: results[s])
print(f"\nbest rotation: {best_shift} with RMS={results[best_shift]:.4f} px")
print(
    "If rotation 0 wins by a wide margin, our object-point convention is\n"
    "correct. If a different rotation wins by orders of magnitude, our\n"
    "per-tag corner order needs to be cycled by that many positions."
)
