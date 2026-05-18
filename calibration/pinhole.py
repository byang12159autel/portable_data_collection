#!/usr/bin/env python3
"""Pinhole intrinsics calibration on an already-unwarped video.

Use this when the source is *not* a raw fisheye lens — e.g. Insta360 Studio /
mobile app exports labelled "single lens (unwarped)", which apply the camera's
own dewarp and emit a roughly-pinhole MP4. The two-stage script's equidistant
unwrap would re-warp such a frame; this one skips Stage 1 and runs the
standard ``cv2.calibrateCamera`` directly.

Output is a `pinhole_intrinsics.npz` carrying ``K`` (3x3), ``D`` (5,) and
``image_size`` — the same shape as Stage 2 of `two_stage_calibrate.py` but
without the fisheye/pinhole_rough entries that don't apply here.

Usage::

    pixi run python -m calibration.pinhole \\
        --video data/insta360_calibration/app_export_singlelens.mp4 \\
        --marker-config config/apriltag_board.yaml \\
        --output data/insta360_calibration/pinhole_intrinsics.npz
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path  # noqa: TC003 — tyro needs Path at runtime
from typing import Literal

import cv2
import numpy as np
import tyro

from core.markers import (
    apriltag_grid_object_points,
    charuco_object_points,
    create_charuco_board,
    detect_aruco_markers,
    detect_charuco_corners,
    load_apriltag_grid_configs,
    load_charuco_board_configs,
)

TargetType = Literal["charuco", "apriltag_grid"]


@dataclasses.dataclass
class Args:
    """Pinhole intrinsics on an unwarped video."""

    video: Path
    """Input video, already projected to a pinhole-like image."""

    marker_config: Path
    """YAML with a ``charuco:`` or ``apriltag_grid:`` section."""

    output: Path
    """Destination .npz."""

    target_type: TargetType = "apriltag_grid"
    """charuco | apriltag_grid."""

    frame_stride: int = 2
    """Use every Nth frame."""

    min_corners_per_view: int = 8
    """Drop a view with fewer than this many detected corners."""

    min_views: int = 10
    """Minimum frames with detections to attempt calibration."""

    fix_k3: bool = True
    """Hold k3 at zero. High-order radial distortion needs corners at large
    image radii to be identifiable; with marginal coverage, freeing k3
    overfits."""

    fix_aspect_ratio: bool = True
    """Enforce fx == fy (square pixels). Sensor pixels on the X4/X5 unwarp
    are square, and without this the focal-length pair is poorly constrained
    unless the board is captured with many oblique tilts."""

    rational_model: bool = False
    """Use the 8-coefficient rational distortion model (k1..k6 + p1, p2)
    instead of the standard 5-coef one. The Insta360 mobile app's single-lens
    unwarp isn't a true pinhole; the extra coefficients capture its residual
    barrel/pincushion behaviour."""

    tag_corner_shift: Literal[0, 1, 2, 3] = 0
    """Cyclically shift each detected tag's 4 corners by this many positions
    before pairing with obj_pts. Use when the physical board's tags are
    mounted in a non-Kalibr orientation (e.g. the Insta360 mobile app's
    'single lens (unwarped)' export ends up with tags rotated 180 deg vs the
    Kalibr +y-up convention — set to 2). Sweep 0/1/2/3 and pick the value
    that produces the lowest per-frame solvePnP residual (see
    debug_pinhole_sanity.py)."""


def _build_extractor(args: Args):
    """Return an ``extract(rgb) -> (obj_pts, img_pts) | None`` closure."""
    if args.target_type == "charuco":
        boards = load_charuco_board_configs(args.marker_config)
        if not boards:
            raise RuntimeError(f"no charuco board in {args.marker_config}")
        name, cfg = next(iter(boards.items()))
        board = create_charuco_board(cfg)
        print(f"Target: ChArUco '{name}' ({cfg.squares_x}x{cfg.squares_y})")

        def extract(rgb: np.ndarray):
            ch_corners, ch_ids, _, _ = detect_charuco_corners(
                rgb, board, min_corners=args.min_corners_per_view
            )
            if ch_corners is None or ch_ids is None:
                return None
            obj = charuco_object_points(board, ch_ids).astype(np.float32).reshape(-1, 1, 3)
            img = ch_corners.reshape(-1, 1, 2).astype(np.float32)
            return obj, img

        return extract

    if args.target_type == "apriltag_grid":
        grids = load_apriltag_grid_configs(args.marker_config)
        if not grids:
            raise RuntimeError(f"no apriltag_grid in {args.marker_config}")
        name, grid_cfg = next(iter(grids.items()))
        dict_id = grid_cfg.cv2_dictionary
        allowed_ids = set(grid_cfg.tag_ids)
        grid_obj_pts = apriltag_grid_object_points(grid_cfg)
        print(
            f"Target: AprilGrid '{name}' ({grid_cfg.tag_cols}x{grid_cfg.tag_rows}, "
            f"{grid_cfg.dictionary})"
        )

        shift = args.tag_corner_shift

        def extract(rgb: np.ndarray):
            corners, ids = detect_aruco_markers(
                rgb, marker_dict=dict_id, allowed_ids=allowed_ids
            )
            if ids is None:
                return None
            obj_list: list[np.ndarray] = []
            img_list: list[np.ndarray] = []
            for c, tid in zip(corners, ids.flatten()):
                tid = int(tid)
                if tid not in grid_obj_pts:
                    continue
                obj_pts_tag = grid_obj_pts[tid]
                if shift:
                    obj_pts_tag = np.roll(obj_pts_tag, -shift, axis=0)
                obj_list.append(obj_pts_tag)
                img_list.append(c.reshape(4, 2))
            if not obj_list or sum(len(p) for p in obj_list) < args.min_corners_per_view:
                return None
            obj = np.vstack(obj_list).astype(np.float32).reshape(-1, 1, 3)
            img = np.vstack(img_list).astype(np.float32).reshape(-1, 1, 2)
            return obj, img

        return extract

    raise ValueError(args.target_type)


def main(args: Args) -> None:
    if not args.video.is_file():
        print(f"Error: {args.video} not found", file=sys.stderr)
        sys.exit(1)

    extract = _build_extractor(args)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {args.video}")
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{args.video}  ({n_total} frames)")

    image_size: tuple[int, int] | None = None
    obj_views: list[np.ndarray] = []
    img_views: list[np.ndarray] = []
    idx = -1
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        idx += 1
        if idx % args.frame_stride != 0:
            continue
        if image_size is None:
            h, w = bgr.shape[:2]
            image_size = (w, h)
            print(f"image size: {w}x{h}  tag_corner_shift={args.tag_corner_shift}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = extract(rgb)
        if res is None:
            continue
        obj_views.append(res[0])
        img_views.append(res[1])
        if idx % 200 == 0:
            print(f"  frame {idx}/{n_total}  views={len(obj_views)}")
    cap.release()

    if image_size is None:
        print("Error: no frames decoded", file=sys.stderr)
        sys.exit(1)

    print(f"collected {len(obj_views)} views")
    if len(obj_views) < args.min_views:
        print(
            f"Error: only {len(obj_views)} views (need >= {args.min_views})",
            file=sys.stderr,
        )
        sys.exit(1)

    flags = 0
    if not args.rational_model:
        flags |= cv2.CALIB_ZERO_TANGENT_DIST  # p1 = p2 = 0
        if args.fix_k3:
            flags |= cv2.CALIB_FIX_K3
    else:
        flags |= cv2.CALIB_RATIONAL_MODEL

    free_params = ["f" if args.fix_aspect_ratio else "fx,fy", "cx", "cy"]
    if args.rational_model:
        free_params += ["k1..k6", "p1", "p2"]
    else:
        free_params += ["k1", "k2"]
        if not args.fix_k3:
            free_params.append("k3")

    K_init: np.ndarray | None = None
    if args.fix_aspect_ratio:
        flags |= cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_USE_INTRINSIC_GUESS
        # CALIB_FIX_ASPECT_RATIO locks the *ratio* fx/fy from the initial K,
        # so we have to supply one. Use a reasonable Insta360 single-lens-export
        # guess: HFOV ~80deg => f ~ W / (2 tan(40deg)) ~ 0.6 * W.
        f_init = 0.6 * image_size[0]
        K_init = np.array(
            [[f_init, 0.0, image_size[0] / 2.0],
             [0.0, f_init, image_size[1] / 2.0],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    print(f"running cv2.calibrateCamera (refining: {', '.join(free_params)}) ...")

    rms, K, D, _, _ = cv2.calibrateCamera(
        obj_views,
        img_views,
        image_size,
        K_init,
        None,
        flags=flags,
    )
    rms = float(rms)
    print(f"RMS = {rms:.4f} px")
    print(
        f"K: fx={K[0, 0]:.2f} fy={K[1, 1]:.2f} "
        f"cx={K[0, 2]:.2f} cy={K[1, 2]:.2f}"
    )
    print(f"D: {D.ravel().tolist()}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(args.output),
        K=K.astype(np.float64),
        D=D.ravel().astype(np.float64),
        image_size=np.array(image_size, dtype=np.int32),
        rms=np.array(rms),
        n_views=np.array(len(obj_views)),
    )
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main(tyro.cli(Args))
