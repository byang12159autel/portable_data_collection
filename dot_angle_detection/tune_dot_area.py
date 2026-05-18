#!/usr/bin/env python3
"""Interactive area-threshold picker for ``detect_black_circular_dots``.

Pulls one frame from the source video, undistorts it via ``Lens0Rectifier``
(same pipeline as ``hinge_pipeline.py``), and serves a viser preview at
``http://localhost:<port>`` with:

- Two reference circles drawn on the image — green for the *min* area,
  red for the *max* area. Move them around with the X/Y sliders to park
  them over real objects (e.g. a chopstick dot or a stray speck) and
  compare visually.
- A live-detector overlay (toggle on/off) that runs
  ``detect_black_circular_dots`` with the current sliders and outlines
  each surviving contour. Read the count + per-detection areas in the
  info panel.

Slide the area sliders until the green circle is just smaller than the
chopstick dots and the red is just bigger; then plug those values into
``hinge_pipeline.py`` via ``--dot-min-area`` / ``--dot-max-area``.

Usage::

    pixi run python -m dot_angle_detection.tune_dot_area \\
        --video data/aruco_test/VID_20260517_192400_00_009.insv \\
        --intrinsics data/insta360_calibration/lens0_combined_subpixel_best.npz \\
        --frame 365
"""

from __future__ import annotations

import dataclasses
import math
import time
from pathlib import Path  # noqa: TC003 — tyro needs Path at runtime

import cv2
import numpy as np

from dot_angle_detection.detect_dots import (
    detect_black_circular_dots,
    detect_white_circular_dots,
)
from dot_angle_detection.homography_transform import (
    Lens0Rectifier,
    _detect_marker,
    _load_marker,
    _marker_object_points,
    _resolve_lens0_mp4,
    homography_from_aruco_pose,
)


@dataclasses.dataclass
class Args:
    """Tune min/max area thresholds for ``detect_black_circular_dots``."""

    video: Path
    """Source recording — .insv (auto-demuxed) or per-lens .mp4."""

    intrinsics: Path | None = None
    """Lens0 calibration .npz. If omitted, the frame is shown raw
    (use only when you've already extracted an undistorted mp4)."""

    frame: int = 0
    """Frame index to sample. Default 0 (first frame)."""

    port: int = 8085
    """Viser preview port."""

    initial_min_area: int = 70
    """Starting value for the min-area slider."""

    initial_max_area: int = 700
    """Starting value for the max-area slider."""

    slider_max: int = 10000
    """Upper bound of both area sliders. Increase if your dots are huge."""

    initial_threshold: int = 80
    """Default threshold (used when polarity=='dark'). For 'light' the
    GUI defaults to 175 instead."""

    initial_min_circularity: float = 0.55

    polarity: str = "dark"
    """Which dot pass to tune: ``dark`` for black dots, ``light`` for
    white dots. Live-detection overlay uses the matching detector."""

    detect_in_plane: bool = False
    """If True, run ArUco detection on the frame, warp it to a
    fronto-parallel marker plane, and tune area thresholds on the
    warped grid. Requires ``--marker-config`` and a frame where the
    marker is visible."""

    marker_config: Path | None = None
    """Marker YAML — required when ``--detect-in-plane`` is set."""

    marker_name: str | None = None

    detect_plane_px: int = 800
    detect_plane_extent_factor: float = 4.0

    dot_plane_z_offset_m: float = -0.005
    """Z-offset of the dot plane (metres) relative to the marker face.
    Must match the value used in ``hinge_pipeline.py`` or the warp
    won't slice through the dot plane."""

    force_demux: bool = False


def _format_detection_table(dets: list[dict], limit: int = 8) -> str:
    if not dets:
        return "  (no detections pass current filters)"
    lines = [f"  {len(dets)} detection(s):"]
    for i, d in enumerate(dets[:limit]):
        cx, cy = d["center"]
        lines.append(
            f"    #{i}: area={d['area']:7.1f}  circ={d['circularity']:.2f}  "
            f"@ ({cx:6.1f}, {cy:6.1f})"
        )
    if len(dets) > limit:
        lines.append(f"    ... and {len(dets) - limit} more")
    return "\n".join(lines)


def main(args: Args) -> None:
    # --- Source frame -------------------------------------------------------
    if args.video.suffix.lower() == ".insv":
        lens0_mp4 = _resolve_lens0_mp4(args.video, args.force_demux)
    else:
        lens0_mp4 = args.video

    cap = cv2.VideoCapture(str(lens0_mp4))
    if not cap.isOpened():
        raise SystemExit(f"could not open {lens0_mp4}")
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_idx = max(0, min(args.frame, n_total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read frame {target_idx} from {lens0_mp4}")

    if args.intrinsics is not None:
        rect = Lens0Rectifier.from_npz(args.intrinsics, (src_w, src_h))
        undistorted = rect.apply(bgr)
        K = rect.K
    else:
        undistorted = bgr
        K = None

    if args.detect_in_plane:
        if args.marker_config is None:
            raise SystemExit("--detect-in-plane requires --marker-config")
        if K is None:
            raise SystemExit("--detect-in-plane requires --intrinsics")
        marker_cfg, dict_id = _load_marker(args.marker_config, args.marker_name)
        corners = _detect_marker(undistorted, dict_id, marker_cfg.id)
        if corners is None:
            raise SystemExit(
                f"marker '{marker_cfg.id}' not found in frame {target_idx}; "
                "pick a different --frame or switch to image-space tuning"
            )
        ok_pnp, rvec, tvec = cv2.solvePnP(
            _marker_object_points(marker_cfg.size), corners,
            K, np.zeros(5, dtype=np.float64),
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok_pnp:
            raise SystemExit("solvePnP failed on chosen frame")
        H_plane_to_img, H_img_to_plane = homography_from_aruco_pose(
            K, rvec, tvec, z_offset_m=args.dot_plane_z_offset_m,
        )
        extent_m = marker_cfg.size * args.detect_plane_extent_factor
        px = args.detect_plane_px
        s_meter = px / (2.0 * extent_m)
        H_plane_to_warped = np.array(
            [[s_meter, 0.0, px / 2.0],
             [0.0, -s_meter, px / 2.0],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        H_img_to_warped = H_plane_to_warped @ H_img_to_plane
        frame = cv2.warpPerspective(
            undistorted, H_img_to_warped, (px, px),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        )
        view_label = (
            f"plane warp (frame {target_idx}, "
            f"+-{extent_m*1000:.0f}mm @ {s_meter:.1f}px/mm)"
        )
    else:
        frame = undistorted
        view_label = (
            f"undistorted (frame {target_idx})"
            if args.intrinsics is not None
            else f"raw (frame {target_idx})"
        )
    h, w = frame.shape[:2]
    print(f"loaded {view_label}: {w}x{h}")

    # --- Viser GUI ----------------------------------------------------------
    import viser
    server = viser.ViserServer(port=args.port)
    print(f"viser at http://localhost:{args.port}")

    image_handle = server.gui.add_image(
        np.zeros((h, w, 3), dtype=np.uint8),
        label=view_label,
    )
    min_area_slider = server.gui.add_slider(
        "min area (px^2)", min=1, max=args.slider_max, step=1,
        initial_value=args.initial_min_area,
    )
    max_area_slider = server.gui.add_slider(
        "max area (px^2)", min=1, max=args.slider_max, step=1,
        initial_value=args.initial_max_area,
    )
    cx_slider = server.gui.add_slider(
        "ref circle X", min=0, max=w - 1, step=1, initial_value=w // 2,
    )
    cy_slider = server.gui.add_slider(
        "ref circle Y", min=0, max=h - 1, step=1, initial_value=h // 2,
    )
    polarity_dropdown = server.gui.add_dropdown(
        "polarity",
        options=("dark", "light"),
        initial_value=args.polarity if args.polarity in ("dark", "light") else "dark",
    )
    initial_thr = (
        175 if polarity_dropdown.value == "light" else args.initial_threshold
    )
    threshold_slider = server.gui.add_slider(
        "intensity threshold", min=0, max=255, step=1,
        initial_value=initial_thr,
    )
    circ_slider = server.gui.add_slider(
        "min circularity", min=0.0, max=1.0, step=0.01,
        initial_value=args.initial_min_circularity,
    )
    show_dets = server.gui.add_checkbox(
        "show live detections", initial_value=True,
    )
    info_text = server.gui.add_text("info", initial_value="", disabled=True)

    def render() -> None:
        canvas = frame.copy()
        min_r = max(1, int(round(math.sqrt(min_area_slider.value / math.pi))))
        max_r = max(min_r + 1, int(round(math.sqrt(max_area_slider.value / math.pi))))
        cx, cy = int(cx_slider.value), int(cy_slider.value)

        # Reference circles (green = min, red = max).
        cv2.circle(canvas, (cx, cy), min_r, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), max_r, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.drawMarker(canvas, (cx, cy), (255, 255, 255),
                       markerType=cv2.MARKER_CROSS, markerSize=14, thickness=1)
        cv2.putText(canvas,
                    f"min={min_area_slider.value}px^2 r~={min_r}px",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas,
                    f"max={max_area_slider.value}px^2 r~={max_r}px",
                    (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2, cv2.LINE_AA)

        info_lines = [
            f"min_area = {int(min_area_slider.value)}  (r ~ {min_r}px)",
            f"max_area = {int(max_area_slider.value)}  (r ~ {max_r}px)",
            f"threshold = {int(threshold_slider.value)}  "
            f"min_circularity = {float(circ_slider.value):.2f}",
        ]

        if show_dets.value:
            detector = (
                detect_white_circular_dots
                if polarity_dropdown.value == "light"
                else detect_black_circular_dots
            )
            dets, _mask = detector(
                frame,
                threshold=int(threshold_slider.value),
                min_area=int(min_area_slider.value),
                max_area=int(max_area_slider.value),
                min_circularity=float(circ_slider.value),
            )
            for d in dets:
                cv2.drawContours(canvas, [d["contour"]], -1,
                                 (0, 255, 255), 1, cv2.LINE_AA)
                px, py = int(round(d["center"][0])), int(round(d["center"][1]))
                cv2.circle(canvas, (px, py), 4, (0, 255, 255), -1)
                cv2.putText(canvas, f"{int(d['area'])}",
                            (px + 6, py - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (0, 255, 255), 1, cv2.LINE_AA)
            info_lines.append(_format_detection_table(dets))

        image_handle.image = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        info_text.value = "\n".join(info_lines)

    for slider in (min_area_slider, max_area_slider, cx_slider, cy_slider,
                   threshold_slider, circ_slider):
        slider.on_update(lambda _: render())
    show_dets.on_update(lambda _: render())
    polarity_dropdown.on_update(lambda _: render())

    render()
    print("ready — tune sliders in the browser. Ctrl-C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    import tyro
    main(tyro.cli(Args))
