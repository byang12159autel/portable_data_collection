"""Sample evenly spaced frames from lens0 and report detection on each."""

import cv2
import numpy as np

from pose_calibration.markers.detect import (
    apriltag_grid_object_points,
    detect_aruco_markers,
    load_apriltag_grid_configs,
    draw_aruco_overlay,
)

video = "data/VID_20260515_121341_00_001_lens0.mp4"
grids = load_apriltag_grid_configs("config/apriltag_board.yaml")
_, grid_cfg = next(iter(grids.items()))
dict_id = grid_cfg.cv2_dictionary
allowed = set(grid_cfg.tag_ids)

cap = cv2.VideoCapture(video)
n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"video has {n_total} frames")

samples = np.linspace(0, n_total - 1, 12, dtype=int)
for i, fidx in enumerate(samples):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
    ok, bgr = cap.read()
    if not ok:
        print(f"frame {fidx}: read failed")
        continue
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    corners, ids = detect_aruco_markers(rgb, marker_dict=dict_id, allowed_ids=allowed)
    n_tags = 0 if ids is None else int(len(ids))
    n_corners = 4 * n_tags
    if corners is not None and ids is not None:
        annotated = draw_aruco_overlay(rgb, corners, ids)
        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    else:
        annotated_bgr = bgr
    # tiny down-scaled thumb to keep file sizes reasonable
    thumb = cv2.resize(annotated_bgr, (640, 640))
    out = f"data/debug_timeline_{i:02d}_frame{fidx:05d}.png"
    cv2.imwrite(out, thumb)
    print(f"  {out}: tags={n_tags}, corners={n_corners}")

cap.release()
