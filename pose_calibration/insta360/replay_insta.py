#!/usr/bin/env python3
"""End-to-end Insta360 preview: rectify + detect, in viser.

Reads the two per-lens videos produced by ``insta360.convert``,
undistorts each to a single virtual pinhole using calibrated
intrinsics, runs the configured marker detectors on each rectified
view, and previews both lenses side-by-side with the familiar
Play/Pause + frame slider GUI from ``replay_video``.

If only one lens video is provided, the missing side is rendered as a
black panel.

Usage::

    pixi run python -m pose_calibration.insta360.replay_insta \\
        --front-video data/scene_lens0.mp4 \\
        --back-video data/scene_lens1.mp4 \\
        --intrinsics data/insta360_intrinsics.npz \\
        --target-type multi \\
        --marker-configs config/aruco_set.yaml config/apriltag_board.yaml
"""

from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path  # noqa: TC003 — tyro needs Path at runtime
from typing import Callable, Literal

import cv2
import numpy as np
import tyro
import viser

from pose_calibration.insta360.rectify import IntrinsicsBundle, Rectifier
from pose_calibration.replay_video import _build_detector_fns

TargetType = Literal["aruco", "charuco", "apriltag", "apriltag_grid", "multi"]
DetectFn = Callable[[np.ndarray], tuple[np.ndarray, int]]


@dataclasses.dataclass
class Args:
    """Rectify + detect Insta360 per-lens videos, preview in viser."""

    intrinsics: Path
    """Per-lens fisheye calibration .npz from insta360.calibrate."""

    front_video: Path | None = None
    """Per-lens recording of the front lens."""

    back_video: Path | None = None
    """Per-lens recording of the back lens."""

    target_type: TargetType = "apriltag_grid"
    """aruco | charuco | apriltag | apriltag_grid | multi."""

    tag_dictionary: str = "DICT_APRILTAG_36h11"
    """OpenCV aruco dictionary for loose `apriltag` mode."""

    marker_configs: tuple[Path, ...] = ()
    """Marker preset YAMLs."""

    fov_deg: float = 110.0
    """Virtual pinhole FOV (degrees)."""

    out_width: int = 960
    """Per-lens preview width."""

    out_height: int = 960
    """Per-lens preview height."""

    port: int = 8085
    """Viser web viewer port."""

    fps: float = 0.0
    """Override playback FPS. 0 = use the source's native FPS."""

    loop: bool = True
    """Loop playback when the video ends."""


@dataclasses.dataclass
class _DetectorArgs:
    """Subset of replay_video.Args that _build_detector_fns reads."""

    target_type: str = "apriltag_grid"
    tag_dictionary: str = "DICT_APRILTAG_36h11"
    marker_configs: tuple[Path, ...] = ()


def _build_detectors(args: Args) -> list[DetectFn]:
    detector_args = _DetectorArgs(
        target_type=args.target_type,
        tag_dictionary=args.tag_dictionary,
        marker_configs=args.marker_configs,
    )
    return _build_detector_fns(detector_args)


def _label_lens(img_bgr: np.ndarray, text: str) -> np.ndarray:
    cv2.putText(
        img_bgr, text, (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA,
    )
    cv2.putText(
        img_bgr, text, (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA,
    )
    return img_bgr


def _open(path: Path) -> tuple[cv2.VideoCapture, int, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    return cap, max(n, 1), fps


def main(args: Args) -> None:
    if args.front_video is None and args.back_video is None:
        print("Error: at least one of --front-video / --back-video is required", file=sys.stderr)
        sys.exit(1)

    bundle = IntrinsicsBundle.load(args.intrinsics)
    out_size = (args.out_width, args.out_height)

    lens_state: dict[str, tuple[cv2.VideoCapture, Rectifier, int, float]] = {}
    for lens, video in [("front", args.front_video), ("back", args.back_video)]:
        if video is None:
            continue
        rectifier = bundle.rectifier_for(lens, out_size, args.fov_deg)
        cap, n, fps = _open(video)
        lens_state[lens] = (cap, rectifier, n, fps)
        print(f"[{lens}] {video}  ({n} frames @ {fps:.1f} FPS)")

    n_frames = min(state[2] for state in lens_state.values())
    native_fps = min(state[3] for state in lens_state.values())
    play_fps = args.fps if args.fps > 0 else native_fps
    frame_dt = 1.0 / play_fps

    detect_fns = _build_detectors(args)
    print(f"Detectors active: {len(detect_fns)}")
    print(
        f"Rectifying each lens -> pinhole {out_size[0]}x{out_size[1]} @ "
        f"{args.fov_deg:.1f} deg FOV"
    )

    def detect_and_draw(rgb: np.ndarray) -> tuple[np.ndarray, int]:
        n_total = 0
        for fn in detect_fns:
            rgb, k = fn(rgb)
            n_total += k
        return rgb, n_total

    server = viser.ViserServer(port=args.port)
    image_handle = server.gui.add_image(
        np.zeros((args.out_height, args.out_width * 2, 3), dtype=np.uint8),
        label="Front | Back",
    )
    play_btn = server.gui.add_checkbox("Playing", initial_value=True)
    frame_slider = server.gui.add_slider(
        "Frame", min=0, max=n_frames - 1, step=1, initial_value=0
    )
    detected_display = server.gui.add_text("Detected", initial_value="0", disabled=True)

    pending_seek: int | None = None
    muted_slider = False

    @frame_slider.on_update
    def _(_: object) -> None:
        nonlocal pending_seek
        if not muted_slider:
            pending_seek = int(frame_slider.value)

    print(f"Viser preview at http://localhost:{args.port}")

    black = np.zeros((args.out_height, args.out_width, 3), dtype=np.uint8)
    last_frame_time = time.time()
    frame_idx = 0

    try:
        while True:
            now = time.time()

            if pending_seek is not None:
                for cap, _r, _n, _fps in lens_state.values():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, pending_seek)
                pending_seek = None
                should_read = True
            elif play_btn.value and (now - last_frame_time) >= frame_dt:
                should_read = True
            else:
                should_read = False

            if should_read:
                panels: dict[str, np.ndarray] = {}
                counts: dict[str, int] = {}
                end_reached = False
                for lens, (cap, rectifier, _n, _fps) in lens_state.items():
                    ok, bgr = cap.read()
                    if not ok:
                        end_reached = True
                        break
                    rgb = cv2.cvtColor(rectifier.apply(bgr), cv2.COLOR_BGR2RGB)
                    annotated, k = detect_and_draw(rgb)
                    _label_lens(annotated, lens.upper())
                    panels[lens] = annotated
                    counts[lens] = k

                if end_reached:
                    if args.loop:
                        for cap, _r, _n, _fps in lens_state.values():
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    play_btn.value = False
                    time.sleep(0.05)
                    continue

                front_panel = panels.get("front", black)
                back_panel = panels.get("back", black)
                preview = np.concatenate([front_panel, back_panel], axis=1)
                image_handle.image = preview
                n_f = counts.get("front", 0)
                n_b = counts.get("back", 0)
                detected_display.value = f"{n_f + n_b}  (front={n_f}, back={n_b})"

                any_cap = next(iter(lens_state.values()))[0]
                frame_idx = max(int(any_cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1, 0)
                muted_slider = True
                frame_slider.value = frame_idx
                muted_slider = False

                last_frame_time = now
            else:
                time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    finally:
        for cap, _r, _n, _fps in lens_state.values():
            cap.release()


if __name__ == "__main__":
    main(tyro.cli(Args))
