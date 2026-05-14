#!/usr/bin/env python3
"""Replay a video file with live marker-detection overlays in viser.

Reads a video, runs one or more marker detectors per frame, and serves a
viser preview at ``http://localhost:<port>`` with detected markers
outlined. Includes Play/Pause and a frame-position slider.

Target types:
  - ``aruco``           Individual ArUco markers (from ``markers:`` YAML)
  - ``charuco``         ChArUco board (from ``charuco:`` YAML)
  - ``apriltag``        Loose AprilTag detection; no YAML needed
  - ``apriltag_grid``   calib.io / Kalibr AprilGrid (from ``apriltag_grid:`` YAML)
  - ``multi``           Run several detectors at once; one detector per
                        section per file across all ``--marker-configs``

Usage::

    # AprilGrid (default)
    python -m pose_calibration.replay_video \\
        --video data/iphone_charuco_test.mov \\
        --marker-configs config/apriltag_board.yaml

    # ArUco markers
    python -m pose_calibration.replay_video --video <path> \\
        --target-type aruco --marker-configs config/aruco_set.yaml

    # Detect both ArUco markers AND AprilGrid in the same frame
    python -m pose_calibration.replay_video --video <path> \\
        --target-type multi \\
        --marker-configs config/aruco_set.yaml config/apriltag_board.yaml
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import Callable, Literal

import cv2
import numpy as np
import tyro
import viser
import yaml

from pose_calibration.detect_marker import (
    create_charuco_board,
    detect_aruco_markers,
    detect_charuco_corners,
    draw_aruco_overlay,
    draw_charuco_overlay,
    load_apriltag_grid_configs,
    load_charuco_board_configs,
    load_marker_configs,
    resolve_aruco_dict,
)

TargetType = Literal["aruco", "charuco", "apriltag", "apriltag_grid", "multi"]

DetectFn = Callable[[np.ndarray], tuple[np.ndarray, int]]


@dataclasses.dataclass
class Args:
    """Video replay with marker-detection overlay."""

    video: Path
    """Path to the video file."""

    target_type: TargetType = "apriltag_grid"
    """aruco | charuco | apriltag | apriltag_grid | multi."""

    tag_dictionary: str = "DICT_APRILTAG_36h11"
    """OpenCV aruco dictionary for loose `apriltag` mode (other modes ignore this)."""

    marker_configs: tuple[Path, ...] = ()
    """Marker preset YAML paths. Required for everything except `apriltag`.
    Pass one for aruco/charuco/apriltag_grid, one or more for multi."""

    port: int = 8085
    """Viser web viewer port."""

    fps: float = 0.0
    """Override playback FPS. 0 = use the video's native FPS."""

    loop: bool = True
    """Loop playback when the video ends."""


def _open_capture(path: Path) -> tuple[cv2.VideoCapture, int, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    return cap, max(n_frames, 1), fps


# ---------------------------------------------------------------------------
# Detector factories — one DetectFn per (config-section, mode)
# ---------------------------------------------------------------------------


def _make_aruco_detector(config_path: Path) -> DetectFn:
    markers = load_marker_configs(config_path)
    if not markers:
        raise RuntimeError(f"No markers defined in {config_path}")
    by_dict: dict[str, set[int]] = {}
    for mc in markers.values():
        by_dict.setdefault(mc.dictionary, set()).add(mc.id)
    if len(by_dict) > 1:
        raise RuntimeError(
            f"{config_path}: markers use multiple dictionaries ({list(by_dict)}); "
            "list each in a separate YAML and pass both via --marker-configs."
        )
    dict_name, allowed_ids = next(iter(by_dict.items()))
    dict_id = resolve_aruco_dict(dict_name)
    print(f"ArUco ({config_path.name}, {dict_name}, IDs {sorted(allowed_ids)})")

    def detect(rgb: np.ndarray) -> tuple[np.ndarray, int]:
        corners, ids = detect_aruco_markers(
            rgb, marker_dict=dict_id, allowed_ids=allowed_ids
        )
        if corners is not None:
            return draw_aruco_overlay(rgb, corners, ids), int(len(ids))
        return rgb, 0

    return detect


def _make_charuco_detector(config_path: Path) -> DetectFn:
    boards = load_charuco_board_configs(config_path)
    if not boards:
        raise RuntimeError(f"No ChArUco boards defined in {config_path}")
    name, board_cfg = next(iter(boards.items()))
    board = create_charuco_board(board_cfg)
    print(
        f"ChArUco ({config_path.name}, '{name}', "
        f"{board_cfg.squares_x}x{board_cfg.squares_y} squares)"
    )

    def detect(rgb: np.ndarray) -> tuple[np.ndarray, int]:
        ch_corners, ch_ids, mkr_corners, mkr_ids = detect_charuco_corners(rgb, board)
        if ch_corners is not None:
            return (
                draw_charuco_overlay(rgb, ch_corners, ch_ids, mkr_corners, mkr_ids),
                int(len(ch_ids)),
            )
        return rgb, 0

    return detect


def _make_apriltag_grid_detector(config_path: Path) -> DetectFn:
    grids = load_apriltag_grid_configs(config_path)
    if not grids:
        raise RuntimeError(f"No apriltag_grid defined in {config_path}")
    name, grid_cfg = next(iter(grids.items()))
    allowed_ids = set(grid_cfg.tag_ids)
    dict_id = grid_cfg.cv2_dictionary
    last_id = grid_cfg.start_id + len(allowed_ids) - 1
    print(
        f"AprilGrid ({config_path.name}, '{name}', "
        f"{grid_cfg.tag_cols}x{grid_cfg.tag_rows} {grid_cfg.dictionary}, "
        f"IDs {grid_cfg.start_id}..{last_id})"
    )

    def detect(rgb: np.ndarray) -> tuple[np.ndarray, int]:
        corners, ids = detect_aruco_markers(
            rgb, marker_dict=dict_id, allowed_ids=allowed_ids
        )
        if corners is not None:
            return draw_aruco_overlay(rgb, corners, ids), int(len(ids))
        return rgb, 0

    return detect


def _make_apriltag_dict_detector(dict_name: str) -> DetectFn:
    dict_id = resolve_aruco_dict(dict_name)
    print(f"AprilTag (loose, {dict_name})")

    def detect(rgb: np.ndarray) -> tuple[np.ndarray, int]:
        corners, ids = detect_aruco_markers(rgb, marker_dict=dict_id)
        if corners is not None:
            return draw_aruco_overlay(rgb, corners, ids), int(len(ids))
        return rgb, 0

    return detect


def _detectors_from_config(path: Path) -> list[DetectFn]:
    """Auto-build a detector for each known top-level section in *path*."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    fns: list[DetectFn] = []
    if raw.get("markers"):
        fns.append(_make_aruco_detector(path))
    if raw.get("apriltag_grid"):
        fns.append(_make_apriltag_grid_detector(path))
    if raw.get("charuco"):
        fns.append(_make_charuco_detector(path))
    if not fns:
        raise RuntimeError(
            f"{path}: no recognised sections (markers, apriltag_grid, charuco)"
        )
    return fns


def _build_detector_fns(args: Args) -> list[DetectFn]:
    if args.target_type == "apriltag":
        return [_make_apriltag_dict_detector(args.tag_dictionary)]

    if args.target_type == "multi":
        if not args.marker_configs:
            raise RuntimeError(
                "--marker-configs is required for multi mode (pass one or more YAMLs)"
            )
        fns: list[DetectFn] = []
        for path in args.marker_configs:
            fns.extend(_detectors_from_config(path))
        return fns

    # single-config modes
    if not args.marker_configs:
        raise RuntimeError(
            f"--marker-configs is required for target-type={args.target_type}"
        )
    if len(args.marker_configs) > 1:
        raise RuntimeError(
            f"target-type={args.target_type} expects exactly one --marker-configs "
            f"(got {len(args.marker_configs)}); use target-type=multi to combine."
        )
    path = args.marker_configs[0]
    if args.target_type == "aruco":
        return [_make_aruco_detector(path)]
    if args.target_type == "charuco":
        return [_make_charuco_detector(path)]
    if args.target_type == "apriltag_grid":
        return [_make_apriltag_grid_detector(path)]
    raise ValueError(f"unknown target_type: {args.target_type}")


# ---------------------------------------------------------------------------
# Main playback loop
# ---------------------------------------------------------------------------


def main(args: Args) -> None:
    cap, n_frames, video_fps = _open_capture(args.video)
    play_fps = args.fps if args.fps > 0 else video_fps
    frame_dt = 1.0 / play_fps

    detect_fns = _build_detector_fns(args)
    print(f"Video: {args.video}  ({n_frames} frames @ {video_fps:.1f} FPS)")
    print(f"Detectors active: {len(detect_fns)}")

    def detect_and_draw(rgb: np.ndarray) -> tuple[np.ndarray, int]:
        n_total = 0
        for fn in detect_fns:
            rgb, n = fn(rgb)
            n_total += n
        return rgb, n_total

    server = viser.ViserServer(port=args.port)
    image_handle = server.gui.add_image(
        np.zeros((16, 16, 3), dtype=np.uint8), label="Video"
    )
    play_btn = server.gui.add_checkbox("Playing", initial_value=True)
    frame_slider = server.gui.add_slider(
        "Frame", min=0, max=n_frames - 1, step=1, initial_value=0
    )
    detected_display = server.gui.add_text(
        "Detected", initial_value="0", disabled=True
    )

    pending_seek: int | None = None
    muted_slider = False

    @frame_slider.on_update
    def _(_: object) -> None:
        nonlocal pending_seek
        if not muted_slider:
            pending_seek = int(frame_slider.value)

    print(f"Viser preview at http://localhost:{args.port}")

    last_frame_time = time.time()
    frame_idx = 0

    try:
        while True:
            now = time.time()

            if pending_seek is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, pending_seek)
                pending_seek = None
                should_read = True
            elif play_btn.value and (now - last_frame_time) >= frame_dt:
                should_read = True
            else:
                should_read = False

            if should_read:
                ok, bgr = cap.read()
                if not ok:
                    if args.loop:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    play_btn.value = False
                    time.sleep(0.05)
                    continue

                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                annotated, n_detected = detect_and_draw(rgb)
                image_handle.image = annotated
                detected_display.value = str(n_detected)

                frame_idx = max(int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1, 0)
                muted_slider = True
                frame_slider.value = frame_idx
                muted_slider = False

                last_frame_time = now
            else:
                time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()


if __name__ == "__main__":
    main(tyro.cli(Args))
