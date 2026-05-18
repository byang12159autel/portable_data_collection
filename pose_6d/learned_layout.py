#!/usr/bin/env python3
"""Camera-pose history for independent ArUco markers, in viser.

Companion to :mod:`pose_6d.known_board` (which expects a
known AprilGrid layout). Here the marker YAML only specifies per-marker
``id``/``size``/``dictionary`` — the inter-marker layout is *learned* from
the recording itself.

Two-pass pipeline:

1. **Layout learning.** Per frame, equidistant-unwrap, detect every marker
   listed in ``--marker-configs``, and run single-tag PnP per marker. The
   lowest-ID marker that's ever seen is the world origin (``--anchor-id``
   overrides). In every frame where the anchor is co-visible with another
   marker, derive ``T_world_marker = inv(T_camera_anchor) @ T_camera_marker``;
   average these across all co-observations (Markley 2007 quaternion mean +
   translation mean) to get a static layout.

2. **Camera tracking.** Per frame, stack obj_pts (transformed into the
   learned world frame) and image pixels for *every* known marker that's
   visible, then one ``cv2.solvePnP`` gives ``T_world_camera``. Robust to
   the anchor falling out of view, as long as some learned marker is in
   frame.

Viser scene mirrors ``pose_history.py``:
  - Each learned marker is a coloured square outline + small triad at its
    learned world pose.
  - Camera trajectory drawn as a Catmull-Rom spline; sparse coordinate
    triads along it; a highlighted triad at the current pose.
  - GUI image panel re-renders the unwrapped frame with detection overlay
    on slider change.

Usage::

    pixi run python -m pose_6d.learned_layout \\
        --video data/aruco_test/VID_..._lens0.mp4 \\
        --intrinsics data/insta360_calibration/lens0_combined_subpixel_best.npz \\
        --marker-configs config/aruco_bench.yaml
"""

from __future__ import annotations

import dataclasses
import math
import time
from pathlib import Path  # noqa: TC003 — tyro needs Path at runtime

import cv2
import numpy as np
import tyro
import viser

from core.markers import MarkerConfig, load_marker_configs
from core.pipeline import Detections
from core.rectify import Rectifier
from pose_6d.estimator import LearnedLayoutEstimator
from pose_6d.layout import (
    LearnedLayout,
    T_to_wxyz_xyz,
    detect_per_marker_pnp,
)


@dataclasses.dataclass
class Args:
    """Camera-pose history for independent ArUco markers."""

    video: Path
    """Raw per-lens fisheye video (output of insta360.convert)."""

    intrinsics: Path
    """Flat pinhole .npz from bench_subpixel.py / pinhole_calibrate.py.
    Carries K, D, pinhole_size, fov_deg used for the Stage-1 unwrap and
    downstream solvePnP."""

    marker_configs: tuple[Path, ...] = ()
    """One or more YAMLs with a ``markers:`` section."""

    anchor_id: int = -1
    """Anchor marker ID (world origin). -1 = pick the lowest ID seen."""

    frame_stride: int = 1
    """Process every Nth frame for PnP. 1 = every frame."""

    coord_frame_stride: int = 30
    """Draw a triad at every Nth recovered pose along the trajectory."""

    axes_length: float = 0.08
    """Camera triad axis length, metres."""

    axes_radius: float = 0.003
    """Camera triad axis radius, metres."""

    display_width: int = 640
    """Width to downsample the GUI image to. 0 = native pinhole size."""

    port: int = 8085


# ---------------------------------------------------------------------------
# Two-pass precompute
# ---------------------------------------------------------------------------


def _build_rectifier(intrinsics: Path, src_w: int, src_h: int):
    d = np.load(str(intrinsics))
    pinhole_size = (int(d["pinhole_size"][0]), int(d["pinhole_size"][1]))
    fov_deg = float(d["fov_deg"])
    f_eq = src_w / math.pi
    K_fish = np.array(
        [[f_eq, 0.0, src_w / 2.0],
         [0.0,  f_eq, src_h / 2.0],
         [0.0,  0.0,  1.0]],
        dtype=np.float64,
    )
    rectifier = Rectifier.build(K_fish, np.zeros(4), pinhole_size, fov_deg)
    return rectifier, d["K"], d["D"], pinhole_size, fov_deg


def _load_all_marker_configs(paths: tuple[Path, ...]) -> dict[int, MarkerConfig]:
    out: dict[int, MarkerConfig] = {}
    for p in paths:
        for cfg in load_marker_configs(p).values():
            if cfg.id in out:
                raise RuntimeError(f"duplicate marker id {cfg.id} across configs")
            out[cfg.id] = cfg
    return out


@dataclasses.dataclass
class _CameraPose:
    frame_idx: int
    T_world_camera: np.ndarray
    n_tags: int
    all_corners: list[np.ndarray]
    all_ids: list[int]


def _precompute(args: Args):
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {args.video}")
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    rectifier, K, D, pinhole_size, fov_deg = _build_rectifier(args.intrinsics, src_w, src_h)
    print(
        f"Video: {args.video}  ({src_w}x{src_h}, {n_total} frames)\n"
        f"Unwrap: {src_w}x{src_h} -> {pinhole_size[0]}x{pinhole_size[1]} @ {fov_deg:.1f} deg"
    )

    cfgs_by_id = _load_all_marker_configs(args.marker_configs)
    if not cfgs_by_id:
        raise SystemExit("no markers in --marker-configs")
    print(f"Marker configs: {sorted((tid, cfg.size, cfg.dictionary) for tid, cfg in cfgs_by_id.items())}")

    # ----- Pass 1: detect + per-marker PnP, gather frame observations -----
    print("Pass 1: detect + per-marker PnP")
    frame_indices: list[int] = []
    observations: list[dict[int, tuple[np.ndarray, np.ndarray]]] = []
    overlay_corners: list[list[np.ndarray]] = []
    overlay_ids: list[list[int]] = []
    t0 = time.time()
    idx = -1
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        idx += 1
        if idx % args.frame_stride != 0:
            continue
        rgb = cv2.cvtColor(rectifier.apply(bgr), cv2.COLOR_BGR2RGB)
        obs, all_corners, all_ids = detect_per_marker_pnp(rgb, cfgs_by_id, K, D)
        frame_indices.append(idx)
        observations.append(obs)
        overlay_corners.append(all_corners)
        overlay_ids.append(all_ids)
        if (len(observations) % 100) == 0:
            n_with = sum(1 for o in observations if o)
            print(f"  frame {idx}/{n_total}  ({n_with} with detections)")
    cap.release()
    n_with_det = sum(1 for o in observations if o)
    print(
        f"  pass 1 done in {time.time() - t0:.1f}s — "
        f"{n_with_det}/{len(observations)} frames have detections"
    )

    # ----- Learn T_world_marker from co-observed samples -----
    layout = LearnedLayout.from_observations(observations, cfgs_by_id, args.anchor_id)
    seen_ids = sorted({tid for o in observations for tid in o})
    print(f"Anchor: marker {layout.anchor_id} (seen ids: {seen_ids})")
    for tid in seen_ids:
        if tid not in layout.T_world_marker:
            print(f"  marker {tid}: never co-visible with anchor — excluded from layout")
            continue
        if tid == layout.anchor_id:
            continue
        t = layout.T_world_marker[tid][:3, 3]
        print(
            f"  marker {tid}: pos=({t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f}) m"
        )

    # ----- Pass 2: multi-tag PnP per frame -> T_world_camera -----
    print("Pass 2: multi-tag PnP -> T_world_camera")
    estimator = LearnedLayoutEstimator(layout, dist_coeffs=D)
    poses: list[_CameraPose] = []
    for fidx, obs, corners, ids in zip(
        frame_indices, observations, overlay_corners, overlay_ids,
    ):
        if not obs:
            continue
        # Build a Detections from the per-marker PnP image points so the
        # estimator can run pooled PnP across every marker in the layout.
        obs_corners = np.stack(
            [img for _T, img in obs.values()], axis=0,
        ).astype(np.float32)
        obs_ids = np.fromiter(obs.keys(), dtype=np.int32)
        dets = Detections(corners=obs_corners, ids=obs_ids)
        pose = estimator(dets, K)
        if pose.T_world_camera is None:
            continue
        poses.append(
            _CameraPose(
                frame_idx=fidx,
                T_world_camera=pose.T_world_camera,
                n_tags=pose.n_inliers,
                all_corners=corners,
                all_ids=ids,
            )
        )
    print(f"  pass 2 done — {len(poses)} camera poses recovered")
    return poses, layout.T_world_marker, cfgs_by_id, rectifier, layout.anchor_id


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


_MARKER_COLOURS = [
    (255, 220,  80),
    ( 80, 220, 255),
    (255,  80, 200),
    (180, 255,  80),
    (255, 150,  80),
]


def _draw_overlay(
    rgb: np.ndarray, corners: list[np.ndarray], ids: list[int]
) -> np.ndarray:
    out = rgb.copy()
    for c, tid in zip(corners, ids):
        pts = c.reshape(-1, 2).astype(np.int32)
        cv2.polylines(out, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        cx, cy = pts.mean(axis=0).astype(int)
        cv2.putText(out, str(int(tid)), (cx - 10, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return out


def main(args: Args) -> None:
    poses, T_world_marker, cfgs_by_id, rectifier, anchor_id = _precompute(args)
    if not poses:
        raise SystemExit("no camera poses recovered")

    server = viser.ViserServer(port=args.port)

    # Static: per-marker outline + small triad at the learned world pose.
    for i, (tid, T_wm) in enumerate(sorted(T_world_marker.items())):
        size = cfgs_by_id[tid].size
        s = size / 2.0
        local = np.array(
            [[-s,  s, 0], [s,  s, 0], [s, -s, 0], [-s, -s, 0]],
            dtype=np.float64,
        )
        world = (T_wm @ np.hstack([local, np.ones((4, 1))]).T).T[:, :3].astype(np.float32)
        segs = np.stack(
            [np.stack([world[j], world[(j + 1) % 4]], axis=0) for j in range(4)],
            axis=0,
        )
        colour = _MARKER_COLOURS[i % len(_MARKER_COLOURS)]
        server.scene.add_line_segments(
            f"/markers/tag{tid:03d}/outline",
            points=segs,
            colors=np.array(colour, dtype=np.uint8),
            line_width=3.0,
        )
        wxyz, xyz = T_to_wxyz_xyz(T_wm)
        server.scene.add_frame(
            f"/markers/tag{tid:03d}/frame",
            wxyz=tuple(float(v) for v in wxyz),
            position=tuple(float(v) for v in xyz),
            axes_length=size * 0.6,
            axes_radius=size * 0.03,
        )
        label_pos = xyz + T_wm[:3, :3] @ np.array([0.0, 0.0, size * 0.6])
        anchor_tag = " (anchor)" if tid == anchor_id else ""
        server.scene.add_label(
            f"/markers/tag{tid:03d}/label",
            text=f"tag{tid}{anchor_tag}",
            position=tuple(float(v) for v in label_pos),
        )

    # Trajectory polyline (straight segments between consecutive poses)
    cam_xyz = np.array([p.T_world_camera[:3, 3] for p in poses], dtype=np.float32)
    if len(cam_xyz) >= 2:
        segs = np.stack([cam_xyz[:-1], cam_xyz[1:]], axis=1)  # (N-1, 2, 3)
        server.scene.add_line_segments(
            "/trajectory",
            points=segs,
            colors=np.array([80, 180, 255], dtype=np.uint8),
            line_width=2.0,
        )

    # Sparse history triads
    for i, p in enumerate(poses):
        if i % args.coord_frame_stride != 0:
            continue
        wxyz, xyz = T_to_wxyz_xyz(p.T_world_camera)
        server.scene.add_frame(
            f"/history/{i:05d}",
            wxyz=tuple(float(v) for v in wxyz),
            position=tuple(float(v) for v in xyz),
            axes_length=args.axes_length * 0.55,
            axes_radius=args.axes_radius * 0.7,
        )

    current_handle = server.scene.add_frame(
        "/current",
        wxyz=(1.0, 0.0, 0.0, 0.0), position=(0.0, 0.0, 0.0),
        axes_length=args.axes_length * 1.6,
        axes_radius=args.axes_radius * 1.6,
    )

    # GUI
    server.gui.add_text("Poses", initial_value=f"{len(poses)} recovered", disabled=True)
    server.gui.add_text("Anchor", initial_value=f"tag{anchor_id}", disabled=True)
    play_btn = server.gui.add_checkbox("Playing", initial_value=True)
    speed = server.gui.add_slider("Frames/sec", min=1, max=60, step=1, initial_value=30)
    idx_slider = server.gui.add_slider(
        "Pose index", min=0, max=len(poses) - 1, step=1, initial_value=0
    )
    pose_info = server.gui.add_text("Info", initial_value="", disabled=True)
    pin_w, pin_h = rectifier.out_size
    disp_w = args.display_width if args.display_width > 0 else pin_w
    disp_h = int(round(disp_w * pin_h / pin_w))
    image_handle = server.gui.add_image(
        np.zeros((disp_h, disp_w, 3), dtype=np.uint8),
        label="Frame (detections in green)",
    )

    pending_seek: int | None = None
    muted = False

    @idx_slider.on_update
    def _(_: object) -> None:
        nonlocal pending_seek
        if not muted:
            pending_seek = int(idx_slider.value)

    actual_port = getattr(server, "get_port", lambda: args.port)()
    print(f"viser preview at http://localhost:{actual_port}")

    # Frame source for the image panel: sequential cap.read() when the next
    # pose is just ahead of where we already are (cheap), real seek only when
    # the jump is large or backwards (HEVC random-access is expensive).
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"could not reopen {args.video}")
    last_decoded = -1  # video frame index of the last frame we decoded

    def fetch(frame_idx: int) -> np.ndarray | None:
        nonlocal last_decoded
        delta = frame_idx - last_decoded
        if delta <= 0 or delta > 12:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        else:
            # Skip-decode the (delta - 1) frames between us and the target.
            for _ in range(delta - 1):
                cap.grab()
        ok, bgr = cap.read()
        if not ok:
            return None
        last_decoded = frame_idx
        return bgr

    cur = 0
    last = time.time()
    try:
        while True:
            now = time.time()
            if pending_seek is not None:
                cur = pending_seek
                pending_seek = None
                should_update = True
            elif play_btn.value and (now - last) >= (1.0 / max(int(speed.value), 1)):
                cur = (cur + 1) % len(poses)
                should_update = True
            else:
                should_update = False

            if should_update:
                p = poses[cur]
                wxyz, xyz = T_to_wxyz_xyz(p.T_world_camera)
                current_handle.wxyz = tuple(float(v) for v in wxyz)
                current_handle.position = tuple(float(v) for v in xyz)
                pose_info.value = (
                    f"frame={p.frame_idx}  tags={p.n_tags}  "
                    f"cam=({xyz[0]:+.3f}, {xyz[1]:+.3f}, {xyz[2]:+.3f}) m"
                )
                bgr = fetch(p.frame_idx)
                if bgr is not None:
                    rect = rectifier.apply(bgr)
                    if (disp_w, disp_h) != (pin_w, pin_h):
                        rect = cv2.resize(rect, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
                    rgb = cv2.cvtColor(rect, cv2.COLOR_BGR2RGB)
                    # Scale overlay corners into the resized image space.
                    if (disp_w, disp_h) != (pin_w, pin_h):
                        sx, sy = disp_w / pin_w, disp_h / pin_h
                        scaled = [c.copy().reshape(-1, 2) * np.array([sx, sy], dtype=np.float32)
                                  for c in p.all_corners]
                        image_handle.image = _draw_overlay(rgb, scaled, p.all_ids)
                    else:
                        image_handle.image = _draw_overlay(rgb, p.all_corners, p.all_ids)
                muted = True
                idx_slider.value = cur
                muted = False
                last = now
            else:
                time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()


if __name__ == "__main__":
    main(tyro.cli(Args))
