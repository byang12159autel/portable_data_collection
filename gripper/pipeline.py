#!/usr/bin/env python3
"""End-to-end hinge-angle pipeline (lens0 -> undistort -> dots -> angle).

Pipeline per frame:

1. **Undistort.** ``core.rectify.Rectifier.from_subpixel_npz`` composes
   the Stage-1 equidistant unwrap + Stage-2 ``cv2.undistort`` from the
   lens0 calibration npz into a single remap LUT. Output is a clean
   pinhole frame with intrinsics ``K`` and zero distortion.
2. **ArUco pose.** Detect the marker from ``--marker-config`` and run
   ``cv2.solvePnP`` (IPPE_SQUARE). The recovered ``rvec, tvec`` feeds
   ``homography_from_aruco_pose`` to build the plane <-> image
   homographies.
3. **Dot detection.** ``detect_black_circular_dots`` finds the four
   chopstick reference dots. Detections that fall inside the marker
   quad are dropped (those are the marker's own black border). If more
   than four pass, we keep the four closest to the marker (in plane
   coordinates).
4. **Hinge angle in plane coordinates.** The four dot centers are
   transformed through ``H_img_to_plane`` into the marker's plane
   (metres, +X right, +Y up). The hinge is computed in plane coords
   instead of image pixels, so the angle is view-independent: as long
   as the marker is visible, the answer is the true in-plane chopstick
   angle regardless of camera tilt or distance.

Outputs:

- ``<video-stem>_hinge.mp4`` with overlays (marker quad + axes, plane
  grid, dot detections, chopstick segments, per-frame angle).
- Per-frame angle stream optionally streamed to viser at
  ``http://localhost:<port>``.

Usage::

    pixi run python -m gripper.pipeline \\
        --video data/aruco_test/VID_20260517_192400_00_009.insv \\
        --intrinsics data/insta360_calibration/lens0_combined_subpixel_best.npz \\
        --marker-config config/chopsticks-v1.yaml
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path  # noqa: TC003 — tyro needs Path at runtime

import cv2
import numpy as np

from core.camera.convert import resolve_lens0_mp4
from core.geometry import (
    apply_homography,
    apply_homography_batch,
    homography_from_aruco_pose,
)
from core.markers import (
    detect_aruco_markers,
    load_named_marker,
    marker_object_points,
)
from core.rectify import Rectifier
from core.viz.birdseye import embed_inset
from core.viz.overlays import (
    draw_axes_via_homography,
    draw_detection,
    draw_plane_grid,
    draw_z_axis,
)
from gripper.dots import (
    detect_black_circular_dots,
    detect_white_circular_dots,
)
from gripper.hinge import (
    hinge_angle_in_plane,
    run_detection_pass,
    select_four_dots,
)


# --------------------------------------------------------------------------
# Per-frame drawing helpers (pure math lives in gripper.hinge).
# --------------------------------------------------------------------------


def _draw_chopstick_overlay(
    canvas: np.ndarray,
    ordered_plane: dict[str, np.ndarray],
    H_plane_to_img: np.ndarray,
) -> None:
    """Draw the two chopstick line segments + endpoint labels in image space.

    ``ordered_plane`` carries plane-coord endpoints; we warp them back into
    pixel space through the homography so labels and lines line up with the
    actual dots even when the marker is heavily tilted.
    """
    palette = {
        "left_top": (255, 80, 80),
        "left_bottom": (255, 80, 80),
        "right_top": (80, 80, 255),
        "right_bottom": (80, 80, 255),
    }
    pix = {
        name: tuple(int(v) for v in apply_homography(H_plane_to_img, pt))
        for name, pt in ordered_plane.items()
    }
    # Vector lines that connect the top/bottom dot of each chopstick.
    # Uncomment when you want them back on the main canvas.
    # cv2.line(canvas, pix["left_bottom"], pix["left_top"],
    #          (255, 80, 80), 2, cv2.LINE_AA)
    # cv2.line(canvas, pix["right_bottom"], pix["right_top"],
    #          (80, 80, 255), 2, cv2.LINE_AA)
    for name, p in pix.items():
        cv2.circle(canvas, p, 6, palette[name], 2)
        cv2.putText(canvas, name.replace("_", "\n"),
                    (p[0] + 8, p[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    palette[name], 1, cv2.LINE_AA)


def _birdseye_with_dots(
    img: np.ndarray,
    H_plane_to_img: np.ndarray,
    centers_plane: np.ndarray | None,
    ordered_plane: dict[str, np.ndarray] | None,
    extent_m: float,
    width_px: int,
    height_px: int,
    angle_deg: float | None,
    extra_dots: list[tuple[np.ndarray, tuple[int, int, int]]] | None = None,
) -> np.ndarray:
    """Top-down warp of the marker plane with dots + chopstick segments.

    ``extra_dots`` is an optional list of ``(centers_plane, BGR)`` pairs
    drawn on top of the warped view (e.g. black-dot detections shown
    alongside the 4 white dots used for the angle).
    """
    s = width_px / (2.0 * extent_m)
    extent_y_m = height_px / (2.0 * s)
    H_plane_to_out = np.array(
        [[s, 0.0, width_px / 2.0],
         [0.0, -s, height_px / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    H_img_to_out = H_plane_to_out @ np.linalg.inv(H_plane_to_img)
    out = cv2.warpPerspective(
        img, H_img_to_out, (width_px, height_px),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
    )
    # Plane axes + label.
    cx, cy = width_px // 2, height_px // 2
    cv2.line(out, (0, cy), (width_px, cy), (255, 200, 80), 1)
    cv2.line(out, (cx, 0), (cx, height_px), (255, 200, 80), 1)
    cv2.arrowedLine(out, (cx, cy), (cx + int(s * extent_m * 0.5), cy),
                    (0, 0, 255), 2, tipLength=0.15)
    cv2.arrowedLine(out, (cx, cy), (cx, cy - int(s * extent_y_m * 0.5)),
                    (0, 255, 0), 2, tipLength=0.15)
    cv2.putText(out,
                f"birds-eye  x+-{extent_m*1000:.0f}mm  y+-{extent_y_m*1000:.0f}mm",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    if centers_plane is not None and len(centers_plane) > 0:
        pix_dots = apply_homography_batch(H_plane_to_out, centers_plane).astype(int)
        for (x, y) in pix_dots:
            cv2.circle(out, (int(x), int(y)), 5, (0, 255, 255), -1)
    if extra_dots:
        for centers, color in extra_dots:
            if centers is None or len(centers) == 0:
                continue
            pix = apply_homography_batch(H_plane_to_out, centers).astype(int)
            for (x, y) in pix:
                cv2.circle(out, (int(x), int(y)), 4, color, -1)
    # Vector lines connecting top/bottom dot of each chopstick in the
    # bird's-eye inset. Uncomment when you want them back.
    # if ordered_plane is not None:
    #     pix = {
    #         n: apply_homography(H_plane_to_out, p).astype(int)
    #         for n, p in ordered_plane.items()
    #     }
    #     cv2.line(out, tuple(pix["left_bottom"]), tuple(pix["left_top"]),
    #              (255, 80, 80), 2, cv2.LINE_AA)
    #     cv2.line(out, tuple(pix["right_bottom"]), tuple(pix["right_top"]),
    #              (80, 80, 255), 2, cv2.LINE_AA)
    if angle_deg is not None:
        cv2.putText(out, f"angle = {angle_deg:6.2f} deg",
                    (8, height_px - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 255), 2, cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------
# CLI / main loop.
# --------------------------------------------------------------------------


@dataclasses.dataclass
class Args:
    """lens0 -> undistort -> ArUco -> dots -> plane-angle pipeline."""

    video: Path
    """Source recording — .insv (auto-demuxed) or per-lens .mp4."""

    intrinsics: Path
    """Lens0 calibration .npz with K, D, pinhole_size, fov_deg."""

    marker_config: Path
    """YAML with a ``markers:`` section (used for pose anchor)."""

    marker_name: str | None = None
    """Marker entry to use inside the YAML. Default: first."""

    output: Path | None = None
    """Output mp4. Defaults to ``<video stem>_hinge.mp4``."""

    port: int = 8085
    """Viser preview port."""

    serve_viser: bool = True
    """Stream a live viser preview while processing."""

    fps: float = 0.0
    """Override output FPS. 0 = source FPS."""

    # --- white dots (used for hinge-angle calc): 4 expected (2 per stick) -

    white_dot_threshold: int = 175
    white_dot_min_area: int = 50
    white_dot_max_area: int = 700
    white_dot_min_circularity: float = 0.55

    # --- black dots (visualization only): 2 expected (1 per stick) --------

    black_dot_threshold: int = 80
    black_dot_min_area: int = 70
    black_dot_max_area: int = 700
    black_dot_min_circularity: float = 0.55

    dot_plane_z_offset_m: float = -0.005
    """Z-offset of the dot plane relative to the marker face, in metres.
    The chopstick dots sit ~5mm behind the marker face (along the
    marker's -Z direction), so the homography is lifted onto a parallel
    plane at this offset. Set to 0 to fall back to the marker plane."""

    detect_in_plane: bool = False
    """If True, warp the frame to a fronto-parallel marker-plane view
    before running dot detection. Removes the ellipse-centre bias and
    makes detection viewpoint-invariant — but ``--dot-min-area`` /
    ``--dot-max-area`` now refer to the warped grid (use
    ``tune_dot_area.py --detect-in-plane`` to re-tune)."""

    detect_plane_px: int = 800
    """Resolution (square) of the plane-detection warp."""

    detect_plane_extent_factor: float = 4.0
    """Plane-detection half-extent as a multiple of marker size."""

    # --- visualization knobs ----------------------------------------------

    axes_length_factor: float = 1.5
    """3D axes length as a multiple of marker size."""

    grid_extent_factor: float = 3.0
    """Plane grid half-extent as a multiple of marker size."""

    grid_step_factor: float = 0.5
    """Plane grid step as a multiple of marker size."""

    birdseye_extent_factor: float = 4.0
    """Bird's-eye half-extent (x) as a multiple of marker size."""

    birdseye_px: int = 320
    """Bird's-eye inset width in pixels."""

    birdseye_height_px: int = 640
    """Bird's-eye inset height in pixels."""

    force_demux: bool = False
    """If --video is .insv, overwrite any existing lens0 mp4."""


def main(args: Args) -> None:
    lens0_mp4 = resolve_lens0_mp4(args.video, args.force_demux)
    marker_cfg, dict_id = load_named_marker(args.marker_config, args.marker_name)
    print(
        f"marker '{args.marker_name or 'default'}': id={marker_cfg.id} "
        f"size={marker_cfg.size} m dict={marker_cfg.dictionary}"
    )

    cap = cv2.VideoCapture(str(lens0_mp4))
    if not cap.isOpened():
        raise SystemExit(f"could not open {lens0_mp4}")
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"video: {lens0_mp4} {src_w}x{src_h} @ {src_fps:.1f} FPS, {n_frames} frames")

    rect = Rectifier.from_subpixel_npz(args.intrinsics, (src_w, src_h))
    out_w, out_h = rect.out_size
    print(f"undistort -> {out_w}x{out_h} pinhole (K fx={rect.K_pinhole[0,0]:.1f})")

    out_path = args.output or args.video.with_name(args.video.stem + "_hinge.mp4")
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
        src_fps if args.fps <= 0 else args.fps, (out_w, out_h),
    )
    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"VideoWriter failed for {out_path}")

    server = None
    image_handle = None
    status_handle = None
    angle_handle = None
    if args.serve_viser:
        try:
            import viser
            server = viser.ViserServer(port=args.port)
            image_handle = server.gui.add_image(
                np.zeros((out_h, out_w, 3), dtype=np.uint8),
                label="lens0 + hinge overlay",
            )
            status_handle = server.gui.add_text(
                "Status", initial_value="starting...", disabled=True,
            )
            angle_handle = server.gui.add_text(
                "Angle (deg)", initial_value="-", disabled=True,
            )
            print(f"viser preview: http://localhost:{args.port}")
        except Exception as e:  # pragma: no cover
            print(f"(viser disabled: {e})")
            server = None

    K = rect.K_pinhole
    distC = np.zeros(5, dtype=np.float64)
    obj_pts = marker_object_points(marker_cfg.size)
    axes_len = marker_cfg.size * args.axes_length_factor
    grid_extent = marker_cfg.size * args.grid_extent_factor
    grid_step = marker_cfg.size * args.grid_step_factor
    birds_extent = marker_cfg.size * args.birdseye_extent_factor

    n_done = 0
    n_pose = 0
    n_angle = 0
    angles_log: list[tuple[int, float]] = []
    t0 = time.time()
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            undistorted = rect.apply(bgr)
            canvas = undistorted.copy()
            status_lines: list[str] = []

            # --- 1. ArUco pose + homography ---------------------------------
            corners_list, _ids = detect_aruco_markers(
                undistorted, marker_dict=dict_id, allowed_ids={marker_cfg.id},
            )
            corners = (
                corners_list[0].reshape(4, 2).astype(np.float32)
                if corners_list else None
            )
            H_plane_to_img = None
            H_img_to_plane = None
            if corners is not None:
                ok_pnp, rvec, tvec = cv2.solvePnP(
                    obj_pts, corners, K, distC,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE,
                )
                if ok_pnp:
                    H_plane_to_img, H_img_to_plane = homography_from_aruco_pose(
                        K, rvec, tvec, z_offset_m=args.dot_plane_z_offset_m,
                    )
                    draw_plane_grid(canvas, H_plane_to_img, grid_extent, grid_step)
                    draw_detection(canvas, corners, marker_cfg.id)
                    draw_axes_via_homography(canvas, H_plane_to_img, axes_len)
                    draw_z_axis(canvas, K, rvec, tvec, axes_len)
                    n_pose += 1
                    status_lines.append(
                        f"d={np.linalg.norm(tvec)*1000:.1f}mm"
                    )

            # --- 2. Dot detection (two passes: white + black) --------------
            # White dots (4 expected, 2 per chopstick) feed the angle
            # calculation; black dots (2 expected, 1 per stick) are
            # detected for visualization only.
            white_dets, white_centers_plane = run_detection_pass(
                undistorted, corners, H_plane_to_img, H_img_to_plane,
                polarity="light",
                detect_in_plane=args.detect_in_plane,
                extent_m=marker_cfg.size * args.detect_plane_extent_factor,
                px=args.detect_plane_px, marker_size=marker_cfg.size,
                threshold=args.white_dot_threshold,
                min_area=args.white_dot_min_area,
                max_area=args.white_dot_max_area,
                min_circularity=args.white_dot_min_circularity,
            )
            black_dets, black_centers_plane = run_detection_pass(
                undistorted, corners, H_plane_to_img, H_img_to_plane,
                polarity="dark",
                detect_in_plane=args.detect_in_plane,
                extent_m=marker_cfg.size * args.detect_plane_extent_factor,
                px=args.detect_plane_px, marker_size=marker_cfg.size,
                threshold=args.black_dot_threshold,
                min_area=args.black_dot_min_area,
                max_area=args.black_dot_max_area,
                min_circularity=args.black_dot_min_circularity,
            )

            # Draw black dots first (visualization only — small red rings).
            for det in black_dets:
                cx, cy = det["center"]
                cv2.circle(canvas, (int(round(cx)), int(round(cy))),
                           6, (60, 60, 240), 2, cv2.LINE_AA)

            # --- 3. Resolve plane coords + 4-dot selection (white only) ----
            ordered_plane = None
            centers_plane: np.ndarray | None = None
            angle_deg = None
            angle_err = None
            if H_img_to_plane is not None and len(white_dets) >= 4:
                if white_centers_plane is None:
                    centers_img = np.array([d["center"] for d in white_dets])
                    white_centers_plane = apply_homography_batch(
                        H_img_to_plane, centers_img
                    )
                white_dets, centers_plane = select_four_dots(
                    white_dets, white_centers_plane
                )

                # Yellow rings around the 4 white dots that feed the angle.
                for det in white_dets:
                    cx, cy = det["center"]
                    cv2.circle(canvas, (int(round(cx)), int(round(cy))),
                               8, (0, 255, 255), 2, cv2.LINE_AA)

                # --- 4. Hinge angle in plane coordinates -------------------
                try:
                    angle_deg, ordered_plane = hinge_angle_in_plane(centers_plane)
                    n_angle += 1
                    angles_log.append((n_done, angle_deg))
                    _draw_chopstick_overlay(canvas, ordered_plane, H_plane_to_img)
                    status_lines.append(
                        f"angle={angle_deg:.2f}deg  "
                        f"W={len(white_dets)}  B={len(black_dets)}"
                    )
                except ValueError as e:
                    angle_err = str(e)
                    status_lines.append(
                        f"angle:fail({e})  W={len(white_dets)}  B={len(black_dets)}"
                    )
            elif H_img_to_plane is not None:
                status_lines.append(
                    f"W={len(white_dets)}  B={len(black_dets)}  (need 4 white)"
                )
            else:
                status_lines.append("no marker")

            # --- Composite + bird's-eye inset ------------------------------
            if H_plane_to_img is not None:
                # If we ran in image-space, fill in black plane coords now.
                if (black_centers_plane is None and len(black_dets) > 0
                        and H_img_to_plane is not None):
                    black_centers_plane = apply_homography_batch(
                        H_img_to_plane,
                        np.array([d["center"] for d in black_dets]),
                    )
                inset = _birdseye_with_dots(
                    undistorted, H_plane_to_img, centers_plane, ordered_plane,
                    birds_extent, args.birdseye_px, args.birdseye_height_px,
                    angle_deg,
                    extra_dots=(
                        [(black_centers_plane, (60, 60, 240))]
                        if black_centers_plane is not None else None
                    ),
                )
                embed_inset(canvas, inset)

            # Header text.
            header_color = (
                (0, 255, 255) if angle_deg is not None
                else (60, 60, 255) if angle_err is not None
                else (200, 200, 200)
            )
            header = f"frame {n_done}  " + "  ".join(status_lines)
            cv2.putText(canvas, header, (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, header_color, 2, cv2.LINE_AA)
            if angle_deg is not None:
                cv2.putText(canvas, f"hinge = {angle_deg:6.2f} deg",
                            (12, out_h - 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 255, 255), 2, cv2.LINE_AA)

            writer.write(canvas)
            n_done += 1

            if image_handle is not None and n_done % 2 == 0:
                image_handle.image = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
                fps_now = n_done / max(time.time() - t0, 1e-3)
                if status_handle is not None:
                    status_handle.value = (
                        f"frame {n_done}/{n_frames}  poses={n_pose}  "
                        f"angles={n_angle}  {fps_now:.1f} fps"
                    )
                if angle_handle is not None:
                    angle_handle.value = (
                        f"{angle_deg:.2f}" if angle_deg is not None
                        else (angle_err or "-")
                    )
            if n_done % 60 == 0:
                fps_now = n_done / max(time.time() - t0, 1e-3)
                last_ang = (
                    f"{angles_log[-1][1]:.2f}deg"
                    if angles_log else "-"
                )
                print(
                    f"  {n_done}/{n_frames}  poses={n_pose}  angles={n_angle}  "
                    f"last={last_ang}  {fps_now:.1f} fps"
                )
    finally:
        cap.release()
        writer.release()

    print(
        f"wrote {out_path} ({n_done} frames, "
        f"{n_pose} with pose, {n_angle} with angle)"
    )
    if angles_log:
        arr = np.array([a for _, a in angles_log])
        print(
            f"angle stats: mean={arr.mean():.2f}  std={arr.std():.2f}  "
            f"min={arr.min():.2f}  max={arr.max():.2f}  deg"
        )
    if server is not None:
        if status_handle is not None:
            status_handle.value = (
                f"done — {n_done} frames, {n_pose} pose, {n_angle} angle. "
                "Press Ctrl-C to exit."
            )
        print("viser still serving; press Ctrl-C to exit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    import tyro
    main(tyro.cli(Args))
