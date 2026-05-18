"""Find a frame with many detected tags and print their layout.

If your physical AprilGrid has tag 0 at the TOP-LEFT (calib.io's default),
but our object points (Kalibr convention) put tag 0 at the BOTTOM-LEFT,
calibration will produce constant ~200 px RMS no matter what. This
script prints the detected layout so you can visually compare to your
printed board.

Usage::

    pixi run python debug_tag_layout.py <video.mp4>
"""

import sys
from pathlib import Path

import cv2

from core.markers import (
    detect_aruco_markers,
    load_apriltag_grid_configs,
)

if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(1)

video = Path(sys.argv[1])
grids = load_apriltag_grid_configs(Path("config/apriltag_board.yaml"))
_, grid_cfg = next(iter(grids.items()))
dict_id = grid_cfg.cv2_dictionary
allowed = set(grid_cfg.tag_ids)

cap = cv2.VideoCapture(str(video))
n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"scanning {n_total} frames for the frame with the most tags ...")

best_ids = None
best_centroids = None
best_n = 0
best_frame_idx = -1
frame_idx = -1
while True:
    ok, bgr = cap.read()
    if not ok:
        break
    frame_idx += 1
    if frame_idx % 10 != 0:  # subsample for speed
        continue
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    corners, ids = detect_aruco_markers(rgb, marker_dict=dict_id, allowed_ids=allowed)
    if ids is None:
        continue
    n = int(len(ids))
    if n > best_n:
        best_n = n
        best_ids = ids.flatten().tolist()
        best_centroids = [c.reshape(4, 2).mean(axis=0).tolist() for c in corners]
        best_frame_idx = frame_idx
cap.release()

if best_ids is None:
    print("no tags detected anywhere")
    sys.exit(1)

print(f"\nbest frame: #{best_frame_idx} with {best_n} tags")
print(f"grid config: {grid_cfg.tag_cols}x{grid_cfg.tag_rows} tags, start_id={grid_cfg.start_id}")
print(f"  expected IDs: {grid_cfg.start_id}..{grid_cfg.start_id + grid_cfg.tag_cols * grid_cfg.tag_rows - 1}")

print("\nDetected tag positions in image (image-y is DOWN):")
print("  id   x_px    y_px    relative position")
for tid, (x, y) in sorted(zip(best_ids, best_centroids)):
    rel_x = "L" if x < 640 else ("R" if x > 1280 else "C")
    rel_y = "T" if y < 640 else ("B" if y > 1280 else "M")
    print(f"  {tid:3d}  {x:7.1f} {y:7.1f}   {rel_y}{rel_x}")

# Specifically: find tag with the lowest visible ID and tag with the highest, report their positions.
low_id_idx = best_ids.index(min(best_ids))
high_id_idx = best_ids.index(max(best_ids))
print(
    f"\nLowest detected ID: {min(best_ids)} at "
    f"({best_centroids[low_id_idx][0]:.0f}, {best_centroids[low_id_idx][1]:.0f})"
)
print(
    f"Highest detected ID: {max(best_ids)} at "
    f"({best_centroids[high_id_idx][0]:.0f}, {best_centroids[high_id_idx][1]:.0f})"
)
print(
    "\nIf you're holding the board face-on:\n"
    "  - calib.io default: tag 0 at TOP-LEFT, IDs increase row-major top-to-bottom\n"
    "  - Kalibr / our code: tag 0 at BOTTOM-LEFT, IDs increase row-major bottom-to-top\n"
    "If the low-ID-tag's image-y is small (top of image) and your board is oriented\n"
    "naturally, you have a calib.io board and our object points are upside-down."
)
