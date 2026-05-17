#!/usr/bin/env python3
"""End-to-end Insta360 preview: rectify + detect, in viser.

Reads the two per-lens videos produced by ``camera.convert``,
undistorts each to a single virtual pinhole using calibrated
intrinsics, runs the configured marker detectors on each rectified
view, and previews both lenses side-by-side with the familiar
Play/Pause + frame slider GUI from ``replay_video``.

If only one lens video is provided, the missing side is rendered as a
black panel.

This app composes a :class:`PreviewPipeline` per lens. Swap a stage by
changing one constructor argument:

  - **Calibration method** -- ``IntrinsicsBundle.load`` already routes
    fisheye, two-stage, and pinhole calibrations through the same
    ``Rectifier`` factory. To use a different rectifier entirely, build
    one satisfying the :class:`Rectifier` protocol and pass it in.
  - **Detection** -- pass any callables matching the ``DrawingDetector``
    signature; ``_build_detector_fns`` produces them from YAML configs.

Usage::

    pixi run python -m pose_calibration.apps.replay_insta \\
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
from typing import Literal

import cv2
import numpy as np
import tyro
import viser

from pose_calibration.apps.replay_video import _build_detector_fns
from pose_calibration.calibration.rectify import IntrinsicsBundle
from pose_calibration.pipeline import PreviewPipeline

TargetType = Literal["aruco", "charuco", "apriltag", "apriltag_grid", "multi"]


@dataclasses.dataclass
class Args:
    """Rectify + detect Insta360 per-lens videos, preview in viser."""

    intrinsics: Path
    """Per-lens fisheye calibration .npz from calibration.fisheye."""

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


def _build_detectors(args: Args) -> list:
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


@dataclasses.dataclass
class _LensFeed:
    """One lens's video capture paired with its PreviewPipeline."""

    cap: cv2.VideoCapture
    pipeline: PreviewPipeline
    n_frames: int
    fps: float


def main(args: Args) -> None:
    if args.front_video is None and args.back_video is None:
        print("Error: at least one of --front-video / --back-video is required", file=sys.stderr)
        sys.exit(1)

    bundle = IntrinsicsBundle.load(args.intrinsics)
    out_size = (args.out_width, args.out_height)
    detect_fns = _build_detectors(args)
    print(f"Detectors active: {len(detect_fns)}")
    print(
        f"Rectifying each lens -> pinhole {out_size[0]}x{out_size[1]} @ "
        f"{args.fov_deg:.1f} deg FOV"
    )

    # Compose one PreviewPipeline per lens. The Rectifier is swappable
    # (currently SinglePinhole via IntrinsicsBundle.rectifier_for); the
    # detect_fns list is swappable (any DrawingDetector callables).
    feeds: dict[str, _LensFeed] = {}
    for lens, video in [("front", args.front_video), ("back", args.back_video)]:
        if video is None:
            continue
        rectifier = bundle.rectifier_for(lens, out_size, args.fov_deg)
        cap, n, fps = _open(video)
        feeds[lens] = _LensFeed(
            cap=cap,
            pipeline=PreviewPipeline(rectifier=rectifier, detect_fns=detect_fns),
            n_frames=n,
            fps=fps,
        )
        print(f"[{lens}] {video}  ({n} frames @ {fps:.1f} FPS)")

    n_frames = min(f.n_frames for f in feeds.values())
    native_fps = min(f.fps for f in feeds.values())
    play_fps = args.fps if args.fps > 0 else native_fps
    frame_dt = 1.0 / play_fps

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
                for f in feeds.values():
                    f.cap.set(cv2.CAP_PROP_POS_FRAMES, pending_seek)
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
                for lens, feed in feeds.items():
                    ok, bgr = feed.cap.read()
                    if not ok:
                        end_reached = True
                        break
                    annotated, k = feed.pipeline.process_bgr(bgr)
                    _label_lens(annotated, lens.upper())
                    panels[lens] = annotated
                    counts[lens] = k

                if end_reached:
                    if args.loop:
                        for f in feeds.values():
                            f.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
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

                any_cap = next(iter(feeds.values())).cap
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
        for f in feeds.values():
            f.cap.release()


if __name__ == "__main__":
    main(tyro.cli(Args))
