"""Try different camera models on the unwarped Insta360 video.

Tests:
  (a) pinhole, 5 coeffs (k1, k2, p1, p2, k3)
  (b) pinhole, rational 8 coeffs (k1..k6, p1, p2)
  (c) fisheye (cv2.fisheye.calibrate, 4 coeffs)

Whichever drives RMS to <~2 px is the projection family the Insta360 app emits.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

from pose_calibration.markers.detect import (
    apriltag_grid_object_points,
    detect_aruco_markers,
    load_apriltag_grid_configs,
)

video = Path(sys.argv[1])
stride = int(sys.argv[2]) if len(sys.argv) > 2 else 4

grids = load_apriltag_grid_configs(Path("config/apriltag_board.yaml"))
_, grid_cfg = next(iter(grids.items()))
grid_obj_pts = apriltag_grid_object_points(grid_cfg)

cap = cv2.VideoCapture(str(video))
n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"{video} ({n_total} frames, stride={stride})")

obj_views, img_views = [], []
size = None
idx = -1
while True:
    ok, bgr = cap.read()
    if not ok:
        break
    idx += 1
    if idx % stride != 0:
        continue
    if size is None:
        h, w = bgr.shape[:2]
        size = (w, h)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    corners, ids = detect_aruco_markers(
        rgb, marker_dict=grid_cfg.cv2_dictionary, allowed_ids=set(grid_cfg.tag_ids)
    )
    if ids is None:
        continue
    obj_list, img_list = [], []
    for c, tid in zip(corners, ids.flatten()):
        tid = int(tid)
        if tid not in grid_obj_pts:
            continue
        obj_list.append(grid_obj_pts[tid])
        img_list.append(c.reshape(4, 2))
    if not obj_list or sum(len(p) for p in obj_list) < 8:
        continue
    obj_views.append(np.vstack(obj_list).astype(np.float32).reshape(-1, 1, 3))
    img_views.append(np.vstack(img_list).astype(np.float32).reshape(-1, 1, 2))

cap.release()
print(f"collected {len(obj_views)} views, image {size}")
assert size is not None

w, h = size
K0 = np.array([[0.6*w, 0, w/2], [0, 0.6*w, h/2], [0, 0, 1]], dtype=np.float64)

# (a) standard pinhole, 5 coeffs, fix aspect
flags_a = cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
rms_a, K_a, D_a, _, _ = cv2.calibrateCamera(obj_views, img_views, size, K0.copy(), None, flags=flags_a)
print(f"\n(a) pinhole 5coef, fix aspect: RMS={rms_a:.3f} px  f={K_a[0,0]:.1f}  D={D_a.ravel().tolist()}")

# (b) rational model 8 coeffs
flags_b = cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_RATIONAL_MODEL
rms_b, K_b, D_b, _, _ = cv2.calibrateCamera(obj_views, img_views, size, K0.copy(), None, flags=flags_b)
print(f"(b) pinhole rational 8coef:    RMS={rms_b:.3f} px  f={K_b[0,0]:.1f}  D={D_b.ravel().tolist()}")

# (c) fisheye
try:
    obj_fe = [o.reshape(1, -1, 3).astype(np.float64) for o in obj_views]
    img_fe = [i.reshape(1, -1, 2).astype(np.float64) for i in img_views]
    K_c = K0.copy()
    D_c = np.zeros(4)
    rms_c, K_c, D_c, _, _ = cv2.fisheye.calibrate(
        obj_fe, img_fe, size, K_c, D_c,
        flags=cv2.fisheye.CALIB_USE_INTRINSIC_GUESS | cv2.fisheye.CALIB_FIX_SKEW,
    )
    print(f"(c) fisheye 4coef:             RMS={rms_c:.3f} px  f={K_c[0,0]:.1f}  D={D_c.ravel().tolist()}")
except Exception as e:
    print(f"(c) fisheye failed: {e}")
