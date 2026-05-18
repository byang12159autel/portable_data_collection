"""Walk every frame of lens0.mp4 and histogram detection quality."""

from collections import Counter

import cv2

from core.markers import (
    detect_aruco_markers,
    load_apriltag_grid_configs,
)

video = "data/VID_20260515_121341_00_001_lens0.mp4"
grids = load_apriltag_grid_configs("config/apriltag_board.yaml")
_, grid_cfg = next(iter(grids.items()))
dict_id = grid_cfg.cv2_dictionary
allowed = set(grid_cfg.tag_ids)

cap = cv2.VideoCapture(video)
n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"scanning {n_total} frames ...")

tag_count_hist = Counter()
spread_buckets = Counter()  # in 0.05 increments
frame = -1

while True:
    ok, bgr = cap.read()
    if not ok:
        break
    frame += 1
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    corners, ids = detect_aruco_markers(rgb, marker_dict=dict_id, allowed_ids=allowed)
    n = 0 if ids is None else int(len(ids))
    tag_count_hist[n] += 1
    if n >= 1 and corners is not None:
        pts = sum((c.reshape(-1, 2).tolist() for c in corners), [])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        spread = min(max(xs) - min(xs), max(ys) - min(ys)) / 1920.0
        spread_buckets[round(spread / 0.05) * 0.05] += 1
    if frame % 500 == 0:
        print(f"  frame {frame}/{n_total}")

cap.release()

print("\ntags detected per frame:")
for k in sorted(tag_count_hist):
    print(f"  {k:>2} tags : {tag_count_hist[k]:>5} frames")

print("\nmin-axis spread fraction (for frames with >=1 tag):")
for k in sorted(spread_buckets):
    print(f"  >={k:.2f} : {spread_buckets[k]:>5} frames")
