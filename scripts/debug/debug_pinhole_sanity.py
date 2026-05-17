"""Sanity-check a single frame from the unwarped Insta360 video:

1. Detect AprilGrid tags
2. solvePnP with a reasonable guess K (no distortion)
3. Reproject and report per-corner pixel error
4. Save an overlay PNG

If the source is a true pinhole the per-corner reproject error should be
< ~2 px. If it's much larger across many tags, the image isn't a true
pinhole — calibration with cv2.calibrateCamera (pinhole model) won't fit.
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
frame_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
f_guess = float(sys.argv[3]) if len(sys.argv) > 3 else 600.0

grids = load_apriltag_grid_configs(Path("config/apriltag_board.yaml"))
_, grid_cfg = next(iter(grids.items()))
grid_obj_pts = apriltag_grid_object_points(grid_cfg)

cap = cv2.VideoCapture(str(video))
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
ok, bgr = cap.read()
cap.release()
if not ok:
    raise SystemExit(f"could not read frame {frame_idx}")

h, w = bgr.shape[:2]
print(f"frame {frame_idx}, image {w}x{h}, f_guess={f_guess}")
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
corners, ids = detect_aruco_markers(
    rgb, marker_dict=grid_cfg.cv2_dictionary, allowed_ids=set(grid_cfg.tag_ids)
)
if ids is None:
    raise SystemExit("no tags detected")

obj_list, img_list, tag_list = [], [], []
for c, tid in zip(corners, ids.flatten()):
    tid = int(tid)
    if tid not in grid_obj_pts:
        continue
    obj_list.append(grid_obj_pts[tid])
    img_list.append(c.reshape(4, 2))
    tag_list.append(tid)

obj_pts = np.vstack(obj_list).astype(np.float32)
img_pts = np.vstack(img_list).astype(np.float32)
print(f"  {len(tag_list)} tags, {len(obj_pts)} corners; tag_ids={sorted(tag_list)}")

K = np.array([[f_guess, 0, w/2], [0, f_guess, h/2], [0, 0, 1]], dtype=np.float64)
ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, np.zeros(5),
                              flags=cv2.SOLVEPNP_ITERATIVE)
print(f"  solvePnP ok={ok}, tvec={tvec.ravel()}")

reproj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, np.zeros(5))
reproj = reproj.reshape(-1, 2)
err = np.linalg.norm(reproj - img_pts, axis=1)
print(f"  reproj err: mean={err.mean():.2f} median={np.median(err):.2f} "
      f"max={err.max():.2f} px (over {len(err)} corners)")

# Per-tag mean error to spot outliers
i = 0
print("  per-tag mean px err:")
for tid in tag_list:
    e = err[i:i+4].mean()
    print(f"    tag {tid:3d}: {e:6.2f}")
    i += 4

overlay = bgr.copy()
for (x, y), (rx, ry) in zip(img_pts, reproj):
    cv2.circle(overlay, (int(x), int(y)), 6, (0, 255, 0), 1)
    cv2.circle(overlay, (int(rx), int(ry)), 4, (0, 0, 255), 2)
    cv2.line(overlay, (int(x), int(y)), (int(rx), int(ry)), (0, 255, 255), 1)
out = video.with_name(video.stem + f"_sanity_f{frame_idx}.png")
cv2.imwrite(str(out), overlay)
print(f"  wrote {out}  (green=detected, red=reprojected w/ f={f_guess})")
