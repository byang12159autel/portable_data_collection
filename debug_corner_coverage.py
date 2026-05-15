"""Plot all detected board corners across every frame onto one image.

A good fisheye calibration recording's coverage plot should look like
confetti filling the entire fisheye circle. A failing one looks like a
small blob in the middle.

Usage::

    pixi run python debug_corner_coverage.py <video.mp4> [<video2.mp4> ...] \\
        [--config config/apriltag_board.yaml] [--out out.png]

Multiple videos accumulate into one combined coverage image (all videos
must share the same frame dimensions). Defaults marker_config to
config/apriltag_board.yaml and the output PNG to
<first-video>_corner_coverage.png (single input) or
<first-video>_combined_corner_coverage.png (multiple inputs).
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from pose_calibration.detect_marker import (
    detect_aruco_markers,
    load_apriltag_grid_configs,
)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("videos", nargs="+", type=Path, help="one or more video files to combine")
parser.add_argument("--config", type=Path, default=Path("config/apriltag_board.yaml"))
parser.add_argument("--out", type=Path, default=None)
args = parser.parse_args()

if args.out is None:
    suffix = "_combined_corner_coverage.png" if len(args.videos) > 1 else "_corner_coverage.png"
    args.out = args.videos[0].with_name(args.videos[0].stem + suffix)

grids = load_apriltag_grid_configs(args.config)
_, grid_cfg = next(iter(grids.items()))
dict_id = grid_cfg.cv2_dictionary
allowed = set(grid_cfg.tag_ids)

heatmap: np.ndarray | None = None
first_frame_bgr: np.ndarray | None = None
shape: tuple[int, int] | None = None  # (h, w)

for video in args.videos:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"could not open {video}")
        sys.exit(1)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"scanning {n_total} frames of {video} ({w}x{h})...")

    if shape is None:
        shape = (h, w)
        heatmap = np.zeros((h, w), dtype=np.uint16)
    elif shape != (h, w):
        print(f"frame size mismatch: {video} is {w}x{h}, expected {shape[1]}x{shape[0]}")
        sys.exit(1)

    frame = -1
    n_with_tags = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frame += 1
        if first_frame_bgr is None:
            first_frame_bgr = bgr.copy()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        corners, ids = detect_aruco_markers(rgb, marker_dict=dict_id, allowed_ids=allowed)
        if ids is None:
            continue
        n_with_tags += 1
        for c in corners:
            for x, y in c.reshape(-1, 2):
                xi, yi = int(round(x)), int(round(y))
                if 0 <= xi < w and 0 <= yi < h:
                    cv2.circle(heatmap, (xi, yi), 6, 1, -1)

    cap.release()
    print(f"  frames with any tag: {n_with_tags}/{n_total}")

assert first_frame_bgr is not None and heatmap is not None
bg = (first_frame_bgr * 0.35).astype(np.uint8)
mask = heatmap > 0
overlay = bg.copy()
overlay[mask] = (60, 255, 60)
cv2.imwrite(str(args.out), overlay)
print(f"wrote {args.out}")
