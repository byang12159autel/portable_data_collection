#!/usr/bin/env python3
"""Composed runner: 6D camera pose + gripper hinge angle on one frame stream.

Wires the two estimators built in the refactor into a single
``RigPipeline``:

  rectify  ->  detect all configured markers  ->  pose_6d + gripper

The world layout for ``pose_6d.LearnedLayoutEstimator`` is learned up
front (pass 1: per-marker PnP across the video). Once the static
``T_world_marker`` dict exists, pass 2 iterates the video again and
runs the composed pipeline per frame, writing:

  - ``<video-stem>_rig.mp4`` — overlay video (marker quads, hinge dots,
    bird's-eye inset, per-frame status text)
  - ``<video-stem>_rig.csv`` — per-frame log: frame_idx, n_pose_inliers,
    cam_xyz (m), cam_qwxyz, angle_deg, n_white, n_black

Optional viser preview at ``http://localhost:<port>``.

Usage::

    pixi run python -m runners.rig_replay \\
        --video data/aruco_test/VID_20260518_093406_00_010.insv \\
        --intrinsics data/insta360_calibration/lens0_combined_subpixel_best.npz \\
        --world-marker-configs config/aruco_bench.yaml \\
        --hinge-marker-config config/chopsticks-v1.yaml
"""

from __future__ import annotations

import csv
import dataclasses
import time
from pathlib import Path  # noqa: TC003 — tyro needs Path at runtime

import cv2
import numpy as np

from core.camera.convert import resolve_lens0_mp4
from core.markers import (
    detect_aruco_markers,
    load_marker_configs,
    load_named_marker,
)
from core.pipeline import Detections, RigPipeline
from core.rectify import Rectifier
from core.viz.birdseye import birdseye, embed_inset
from core.viz.overlays import (
    draw_axes_via_homography,
    draw_detection,
    draw_plane_grid,
)
from gripper.estimator import HingeAngleEstimator
from pose_6d.estimator import LearnedLayoutEstimator
from pose_6d.layout import LearnedLayout, T_to_wxyz_xyz, detect_per_marker_pnp


@dataclasses.dataclass
class Args:
    """Composed pose_6d + gripper runner."""

    video: Path
    """Source recording — ``.insv`` (auto-demuxed) or per-lens ``.mp4``."""

    intrinsics: Path
    """Lens0 sub-pixel calibration ``.npz`` (K, D, pinhole_size, fov_deg)."""

    world_marker_configs: tuple[Path, ...] = ()
    """YAMLs defining the markers fixed in the world for camera-pose tracking.
    Anchor (world origin) is the lowest ID seen unless --anchor-id overrides.
    Not required on a cache hit (the cache is self-contained)."""

    layout_cache: Path | None = None
    """Where to cache the learned world layout. Defaults to
    ``<video-stem>_layout.yaml`` alongside the source video. On a hit
    rig_replay skips pass 1 and loads the layout instead. Pass
    ``--force-relearn`` to bypass an existing cache."""

    force_relearn: bool = False
    """Ignore any existing layout cache and rebuild from pass 1."""

    save_layout: bool = True
    """Persist the freshly-learned layout to ``--layout-cache``. Off when
    you're iterating on the learning step and don't want to overwrite a
    known-good cache."""

    hinge_marker_config: Path | None = None
    """YAML with the marker rigidly attached to the gripper. The hinge plane
    is anchored to this marker. Defaults to ``config/chopsticks-v1.yaml``
    if --hinge-marker-config is omitted."""

    hinge_marker_name: str | None = None
    """Name of the hinge marker inside its YAML. Default: first entry."""

    anchor_id: int = -1
    """World anchor marker ID. -1 = lowest ID seen."""

    output: Path | None = None
    """Output mp4. Defaults to ``<video-stem>_rig.mp4``."""

    output_csv: Path | None = None
    """Per-frame log csv. Defaults to ``<video-stem>_rig.csv``."""

    port: int = 8085
    """Viser preview port."""

    serve_viser: bool = True
    """Stream a live viser preview while processing."""

    learn_stride: int = 1
    """Process every Nth frame during pass 1 (layout learning)."""

    fps: float = 0.0
    """Override output FPS. 0 = source FPS."""

    force_demux: bool = False
    """If --video is .insv, overwrite any existing lens0 mp4."""

    # Hinge dot thresholds — passed through to HingeAngleEstimator.
    dot_plane_z_offset_m: float = -0.005
    detect_in_plane: bool = True
    detect_plane_px: int = 800
    detect_plane_extent_factor: float = 4.0
    white_dot_threshold: int = 175
    white_dot_min_area: int = 50
    white_dot_max_area: int = 700
    white_dot_min_circularity: float = 0.55
    black_dot_threshold: int = 80
    black_dot_min_area: int = 70
    black_dot_max_area: int = 700
    black_dot_min_circularity: float = 0.55

    # Visualization
    axes_length_factor: float = 1.5
    grid_extent_factor: float = 3.0
    grid_step_factor: float = 0.5
    birdseye_extent_factor: float = 4.0
    birdseye_px: int = 320
    birdseye_height_px: int = 640


def _load_world_configs(paths: tuple[Path, ...]) -> dict[int, object]:
    """Merge ``markers:`` from every YAML into one ``{id: MarkerConfig}`` dict."""
    out: dict[int, object] = {}
    for p in paths:
        for cfg in load_marker_configs(p).values():
            if cfg.id in out:
                raise SystemExit(f"duplicate marker id {cfg.id} across world configs")
            out[cfg.id] = cfg
    return out


@dataclasses.dataclass
class _MultiDictDetector:
    """``Detector`` implementation across multiple ArUco dictionaries.

    Each (dictionary, allowed_ids) pair runs as its own
    ``detect_aruco_markers`` call; results are concatenated into a single
    ``Detections``. The world markers + hinge marker often share a
    dictionary, but this handles the case where they don't.
    """

    by_dict: dict[int, set[int]]

    def __call__(self, frame: np.ndarray) -> Detections:
        all_corners: list[np.ndarray] = []
        all_ids: list[int] = []
        for dict_id, allowed in self.by_dict.items():
            corners, ids = detect_aruco_markers(
                frame, marker_dict=dict_id, allowed_ids=allowed,
            )
            if ids is None:
                continue
            for c, tid in zip(corners, ids.flatten()):
                all_corners.append(c.reshape(4, 2).astype(np.float32))
                all_ids.append(int(tid))
        if not all_corners:
            return Detections(
                corners=np.empty((0, 4, 2), dtype=np.float32),
                ids=np.empty((0,), dtype=np.int32),
            )
        return Detections(
            corners=np.stack(all_corners).astype(np.float32),
            ids=np.array(all_ids, dtype=np.int32),
        )


def _learn_layout(
    cap: cv2.VideoCapture,
    rectifier: Rectifier,
    world_cfgs: dict[int, object],
    K: np.ndarray,
    anchor_id: int,
    stride: int,
) -> LearnedLayout:
    """Pass 1: iterate the video, per-marker PnP, build the layout."""
    print("Pass 1: learning world layout")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    observations: list[dict[int, tuple[np.ndarray, np.ndarray]]] = []
    t0 = time.time()
    idx = -1
    D_zero = np.zeros(5, dtype=np.float64)
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        idx += 1
        if idx % stride != 0:
            continue
        rectified = rectifier.apply(bgr)
        obs, _, _ = detect_per_marker_pnp(rectified, world_cfgs, K, D_zero)
        observations.append(obs)
        if (len(observations) % 100) == 0:
            n_with = sum(1 for o in observations if o)
            print(f"  frame {idx}/{n_total}  ({n_with} with detections)")
    print(f"  pass 1 done in {time.time() - t0:.1f}s "
          f"({sum(1 for o in observations if o)}/{len(observations)} with detections)")
    layout = LearnedLayout.from_observations(observations, world_cfgs, anchor_id)
    seen = sorted({tid for o in observations for tid in o})
    print(f"  anchor: marker {layout.anchor_id} (seen ids: {seen})")
    for tid, T in sorted(layout.T_world_marker.items()):
        t = T[:3, 3]
        tag = " (anchor)" if tid == layout.anchor_id else ""
        print(f"    marker {tid}{tag}: pos=({t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f}) m")
    return layout


def _draw_frame_overlay(
    canvas: np.ndarray, detections: Detections, hinge_id: int,
) -> None:
    """Outline every detected marker; the hinge anchor gets a brighter ring."""
    for i, tid in enumerate(np.asarray(detections.ids).flatten().tolist()):
        corners = np.asarray(detections.corners[i]).reshape(4, 2)
        draw_detection(canvas, corners, int(tid))


def main(args: Args) -> None:
    lens0_mp4 = resolve_lens0_mp4(args.video, args.force_demux)

    hinge_yaml = args.hinge_marker_config or Path("config/chopsticks-v1.yaml")
    hinge_cfg, hinge_dict_id = load_named_marker(hinge_yaml, args.hinge_marker_name)
    print(
        f"hinge marker: id={hinge_cfg.id} size={hinge_cfg.size} m "
        f"dict={hinge_cfg.dictionary}"
    )

    cap = cv2.VideoCapture(str(lens0_mp4))
    if not cap.isOpened():
        raise SystemExit(f"could not open {lens0_mp4}")
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    rectifier = Rectifier.from_subpixel_npz(args.intrinsics, (src_w, src_h))
    out_w, out_h = rectifier.out_size
    K = rectifier.K_pinhole
    print(f"video: {lens0_mp4} {src_w}x{src_h} @ {src_fps:.1f} FPS, {n_total} frames")
    print(f"undistort -> {out_w}x{out_h} pinhole (K fx={K[0,0]:.1f})")

    cache_path = args.layout_cache or args.video.with_name(
        args.video.stem + "_layout.yaml"
    )
    if cache_path.exists() and not args.force_relearn:
        print(f"loading world layout from cache: {cache_path}")
        layout = LearnedLayout.load(cache_path)
        world_cfgs = layout.marker_configs
        print(
            "  anchor: marker {layout.anchor_id} "
            f"(cached markers: {sorted(layout.T_world_marker)})".format(layout=layout)
        )
    else:
        world_cfgs = _load_world_configs(args.world_marker_configs)
        if not world_cfgs:
            raise SystemExit(
                "no layout cache found and --world-marker-configs is empty "
                f"(looked for {cache_path}; pass --world-marker-configs or "
                "point --layout-cache at an existing yaml)"
            )
        layout = _learn_layout(cap, rectifier, world_cfgs, K, args.anchor_id, args.learn_stride)
        if args.save_layout:
            layout.save(cache_path)
            print(f"saved layout to {cache_path}")

    print(f"world markers: {sorted((tid, c.size, c.dictionary) for tid, c in world_cfgs.items())}")
    if hinge_cfg.id in world_cfgs:
        print(f"note: hinge marker {hinge_cfg.id} also appears in world layout — "
              "it will be tracked in both branches")

    # Detector covers both branches: world layout ids + hinge anchor id.
    by_dict: dict[int, set[int]] = {}
    for cfg in world_cfgs.values():
        by_dict.setdefault(cfg.cv2_dictionary, set()).add(cfg.id)
    by_dict.setdefault(hinge_dict_id, set()).add(hinge_cfg.id)
    detector = _MultiDictDetector(by_dict)

    pose_est = LearnedLayoutEstimator(layout)
    gripper_est = HingeAngleEstimator(
        marker_id=hinge_cfg.id,
        marker_size=hinge_cfg.size,
        dot_plane_z_offset_m=args.dot_plane_z_offset_m,
        detect_in_plane=args.detect_in_plane,
        detect_plane_px=args.detect_plane_px,
        detect_plane_extent_factor=args.detect_plane_extent_factor,
        white_dot_threshold=args.white_dot_threshold,
        white_dot_min_area=args.white_dot_min_area,
        white_dot_max_area=args.white_dot_max_area,
        white_dot_min_circularity=args.white_dot_min_circularity,
        black_dot_threshold=args.black_dot_threshold,
        black_dot_min_area=args.black_dot_min_area,
        black_dot_max_area=args.black_dot_max_area,
        black_dot_min_circularity=args.black_dot_min_circularity,
    )
    rig = RigPipeline(
        rectifier=rectifier, detector=detector,
        pose_estimator=pose_est, gripper_estimator=gripper_est,
    )

    out_mp4 = args.output or args.video.with_name(args.video.stem + "_rig.mp4")
    out_csv = args.output_csv or args.video.with_name(args.video.stem + "_rig.csv")
    writer = cv2.VideoWriter(
        str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"),
        src_fps if args.fps <= 0 else args.fps, (out_w, out_h),
    )
    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"VideoWriter failed for {out_mp4}")

    server = None
    image_handle = None
    status_handle = None
    angle_handle = None
    pose_handle = None
    if args.serve_viser:
        try:
            import viser
            server = viser.ViserServer(port=args.port)
            image_handle = server.gui.add_image(
                np.zeros((out_h, out_w, 3), dtype=np.uint8),
                label="rig overlay",
            )
            status_handle = server.gui.add_text("Status", initial_value="starting...", disabled=True)
            angle_handle = server.gui.add_text("Angle (deg)", initial_value="-", disabled=True)
            pose_handle = server.gui.add_text("Cam xyz (m)", initial_value="-", disabled=True)
            print(f"viser preview: http://localhost:{args.port}")
        except Exception as e:  # pragma: no cover
            print(f"(viser disabled: {e})")
            server = None

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    grid_extent = hinge_cfg.size * args.grid_extent_factor
    grid_step = hinge_cfg.size * args.grid_step_factor
    axes_len = hinge_cfg.size * args.axes_length_factor
    birds_extent = hinge_cfg.size * args.birdseye_extent_factor

    print("Pass 2: rig pipeline")
    csv_rows: list[list] = []
    n_done = n_pose = n_angle = 0
    t0 = time.time()
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rectified, dets, pose, gripper = rig.process(bgr)
            canvas = rectified.copy()

            _draw_frame_overlay(canvas, dets, hinge_cfg.id)

            if gripper is not None and gripper.H_plane_to_img is not None:
                draw_plane_grid(canvas, gripper.H_plane_to_img, grid_extent, grid_step)
                draw_axes_via_homography(canvas, gripper.H_plane_to_img, axes_len)
                inset = birdseye(
                    rectified, gripper.H_plane_to_img, birds_extent,
                    args.birdseye_px, args.birdseye_height_px,
                )
                embed_inset(canvas, inset)

            # Status text at the top of the canvas.
            status_bits = [f"frame {n_done}/{n_total}"]
            if pose is not None and pose.T_world_camera is not None:
                xyz = pose.T_world_camera[:3, 3]
                status_bits.append(
                    f"cam=({xyz[0]:+.2f},{xyz[1]:+.2f},{xyz[2]:+.2f})m  "
                    f"inliers={pose.n_inliers}"
                )
                n_pose += 1
            else:
                status_bits.append("cam: none")
            if gripper is not None and gripper.angle_deg is not None:
                status_bits.append(
                    f"angle={gripper.angle_deg:.2f}deg  "
                    f"W={gripper.n_white_dots}  B={gripper.n_black_dots}"
                )
                n_angle += 1
            elif gripper is not None:
                status_bits.append(
                    f"angle: -  W={gripper.n_white_dots}  B={gripper.n_black_dots}"
                )
            header_color = (0, 255, 255) if (gripper and gripper.angle_deg is not None) else (200, 200, 200)
            cv2.putText(canvas, "  ".join(status_bits), (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, header_color, 2, cv2.LINE_AA)

            writer.write(canvas)

            # CSV log row.
            row: list = [n_done, pose.n_inliers if pose else 0]
            if pose is not None and pose.T_world_camera is not None:
                xyz = pose.T_world_camera[:3, 3].tolist()
                wxyz = T_to_wxyz_xyz(pose.T_world_camera)[0].tolist()
                row += xyz + wxyz
            else:
                row += [None] * 7
            if gripper is not None:
                row += [gripper.angle_deg, gripper.n_white_dots, gripper.n_black_dots]
            else:
                row += [None, 0, 0]
            csv_rows.append(row)

            n_done += 1
            if image_handle is not None and n_done % 2 == 0:
                image_handle.image = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
                fps_now = n_done / max(time.time() - t0, 1e-3)
                if status_handle is not None:
                    status_handle.value = (
                        f"frame {n_done}/{n_total}  poses={n_pose}  "
                        f"angles={n_angle}  {fps_now:.1f} fps"
                    )
                if angle_handle is not None and gripper is not None:
                    angle_handle.value = (
                        f"{gripper.angle_deg:.2f}" if gripper.angle_deg is not None else "-"
                    )
                if pose_handle is not None and pose is not None and pose.T_world_camera is not None:
                    xyz = pose.T_world_camera[:3, 3]
                    pose_handle.value = f"({xyz[0]:+.3f}, {xyz[1]:+.3f}, {xyz[2]:+.3f})"
            if n_done % 60 == 0:
                fps_now = n_done / max(time.time() - t0, 1e-3)
                print(f"  {n_done}/{n_total}  poses={n_pose}  angles={n_angle}  {fps_now:.1f} fps")
    finally:
        cap.release()
        writer.release()

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "frame_idx", "n_pose_inliers",
            "cam_x", "cam_y", "cam_z", "cam_qw", "cam_qx", "cam_qy", "cam_qz",
            "angle_deg", "n_white_dots", "n_black_dots",
        ])
        w.writerows(csv_rows)

    print(f"wrote {out_mp4} and {out_csv}")
    print(f"summary: {n_done} frames, {n_pose} with pose, {n_angle} with angle")

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
