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

from core.rectify import Rectifier
from core.markers import (
    MarkerConfig,
    detect_aruco_markers,
    load_marker_configs,
    marker_object_points,
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
# Math helpers (copied from pose_history.py for module independence)
# ---------------------------------------------------------------------------


def _rvec_tvec_to_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec.ravel()
    return T


def _R_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> (w, x, y, z) unit quaternion."""
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * math.sqrt(1.0 + tr)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float64)


def _quat_to_R(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
         [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
         [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)]],
        dtype=np.float64,
    )


def _T_to_wxyz_xyz(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return _R_to_quat(T[:3, :3]), T[:3, 3].astype(np.float64)


def _avg_T(Ts: list[np.ndarray]) -> np.ndarray:
    """Mean of 4x4 transforms: arithmetic mean translation, Markley 2007
    quaternion mean for rotation (dominant eigenvector of sum(q qT))."""
    if len(Ts) == 1:
        return Ts[0].copy()
    ts = np.stack([T[:3, 3] for T in Ts])
    qs = np.stack([_R_to_quat(T[:3, :3]) for T in Ts])
    # Hemisphere-align so the eigenvector picks a coherent mean
    for i in range(1, len(qs)):
        if np.dot(qs[0], qs[i]) < 0:
            qs[i] = -qs[i]
    M = qs.T @ qs
    _, V = np.linalg.eigh(M)
    q_avg = V[:, -1]
    T_avg = np.eye(4, dtype=np.float64)
    T_avg[:3, :3] = _quat_to_R(q_avg)
    T_avg[:3, 3] = ts.mean(axis=0)
    return T_avg


# ---------------------------------------------------------------------------
# Detection + PnP
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


def _detect_frame(
    rgb: np.ndarray,
    cfgs_by_id: dict[int, MarkerConfig],
    K: np.ndarray,
    D: np.ndarray,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], list[np.ndarray], list[int]]:
    """Detect all configured markers in *rgb* and run single-tag PnP per marker.

    Returns
    -------
    obs : id -> (T_camera_marker (4x4), img_pts (4,2) float32)
    all_corners : list of detected corners (for overlay)
    all_ids : matching tag ids (for overlay)
    """
    by_dict: dict[int, list[int]] = {}
    for cfg in cfgs_by_id.values():
        by_dict.setdefault(cfg.cv2_dictionary, []).append(cfg.id)

    obs: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    all_corners: list[np.ndarray] = []
    all_ids: list[int] = []
    for dict_id, ids in by_dict.items():
        corners, det_ids = detect_aruco_markers(
            rgb, marker_dict=dict_id, allowed_ids=set(ids)
        )
        if det_ids is None:
            continue
        for c, tid in zip(corners, det_ids.flatten()):
            tid = int(tid)
            cfg = cfgs_by_id[tid]
            img = c.reshape(4, 2).astype(np.float32)
            obj = marker_object_points(cfg.size)
            ok, rvec, tvec = cv2.solvePnP(
                obj, img, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if ok:
                obs[tid] = (_rvec_tvec_to_T(rvec, tvec), img)
            all_corners.append(c)
            all_ids.append(tid)
    return obs, all_corners, all_ids


# ---------------------------------------------------------------------------
# Two-pass precompute
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _FrameObs:
    frame_idx: int
    obs: dict[int, tuple[np.ndarray, np.ndarray]]  # tid -> (T_cam_marker, img_pts)
    all_corners: list[np.ndarray]
    all_ids: list[int]


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
    frame_obs_list: list[_FrameObs] = []
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
        obs, all_corners, all_ids = _detect_frame(rgb, cfgs_by_id, K, D)
        frame_obs_list.append(_FrameObs(idx, obs, all_corners, all_ids))
        if (len(frame_obs_list) % 100) == 0:
            n_with = sum(1 for f in frame_obs_list if f.obs)
            print(f"  frame {idx}/{n_total}  ({n_with} with detections)")
    cap.release()
    n_with_det = sum(1 for f in frame_obs_list if f.obs)
    print(f"  pass 1 done in {time.time() - t0:.1f}s — {n_with_det}/{len(frame_obs_list)} frames have detections")

    # ----- Choose anchor -----
    seen_ids = sorted({tid for f in frame_obs_list for tid in f.obs.keys()})
    if not seen_ids:
        raise SystemExit("no configured markers detected in any frame")
    anchor_id = args.anchor_id if args.anchor_id >= 0 else seen_ids[0]
    if anchor_id not in seen_ids:
        raise SystemExit(
            f"anchor id {anchor_id} never detected; seen: {seen_ids}"
        )
    print(f"Anchor: marker {anchor_id} (seen ids: {seen_ids})")

    # ----- Learn T_world_marker via averaging over co-observed frames -----
    samples: dict[int, list[np.ndarray]] = {tid: [] for tid in seen_ids}
    samples[anchor_id].append(np.eye(4, dtype=np.float64))
    for f in frame_obs_list:
        if anchor_id not in f.obs:
            continue
        T_cam_anchor = f.obs[anchor_id][0]
        T_anchor_cam = np.linalg.inv(T_cam_anchor)
        for tid, (T_cam_m, _) in f.obs.items():
            if tid == anchor_id:
                continue
            samples[tid].append(T_anchor_cam @ T_cam_m)
    T_world_marker: dict[int, np.ndarray] = {}
    for tid, sams in samples.items():
        if not sams:
            print(f"  marker {tid}: never co-visible with anchor — excluded from layout")
            continue
        T_world_marker[tid] = _avg_T(sams)
        if tid != anchor_id:
            t = T_world_marker[tid][:3, 3]
            print(
                f"  marker {tid}: layout from {len(sams)} samples, "
                f"pos=({t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f}) m"
            )

    # ----- Pass 2: multi-tag PnP per frame -> T_world_camera -----
    print("Pass 2: multi-tag PnP -> T_world_camera")
    poses: list[_CameraPose] = []
    for f in frame_obs_list:
        obj_world_list: list[np.ndarray] = []
        img_list: list[np.ndarray] = []
        for tid, (_, img) in f.obs.items():
            if tid not in T_world_marker:
                continue
            obj_local = marker_object_points(cfgs_by_id[tid].size)  # (4, 3)
            obj_h = np.hstack([obj_local, np.ones((4, 1), dtype=np.float64)])
            obj_world = (T_world_marker[tid] @ obj_h.T).T[:, :3].astype(np.float32)
            obj_world_list.append(obj_world)
            img_list.append(img)
        if not obj_world_list:
            continue
        obj_pts = np.vstack(obj_world_list)
        img_pts = np.vstack(img_list)
        if len(obj_pts) < 4:
            continue
        flags = cv2.SOLVEPNP_IPPE_SQUARE if len(obj_pts) == 4 else cv2.SOLVEPNP_ITERATIVE
        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, D, flags=flags)
        if not ok:
            continue
        T_cam_world = _rvec_tvec_to_T(rvec, tvec)
        poses.append(
            _CameraPose(
                frame_idx=f.frame_idx,
                T_world_camera=np.linalg.inv(T_cam_world),
                n_tags=len(obj_world_list),
                all_corners=f.all_corners,
                all_ids=f.all_ids,
            )
        )
    print(f"  pass 2 done — {len(poses)} camera poses recovered")
    return poses, T_world_marker, cfgs_by_id, rectifier, anchor_id


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
        wxyz, xyz = _T_to_wxyz_xyz(T_wm)
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
        wxyz, xyz = _T_to_wxyz_xyz(p.T_world_camera)
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
                wxyz, xyz = _T_to_wxyz_xyz(p.T_world_camera)
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
