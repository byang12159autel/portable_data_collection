#!/usr/bin/env python3
"""Replay a calibration video and visualize camera pose history in viser.

Pipeline per sampled frame:
  raw fisheye -> equidistant unwrap -> 8x upscale + aruco detect (no refine)
  -> apply tag-corner-shift -> pooled solvePnP with calibrated K, D
  -> T_camera_board -> invert -> T_board_camera (camera in board frame)

The viser scene shows:
  - The static AprilGrid board outline at the origin
  - The camera trajectory as a Catmull-Rom spline
  - A sparse triad of coordinate frames along the trajectory
  - A highlighted coordinate frame at the *current* pose (slider-controlled)
  - The unwrapped image with detected tag overlay, in a GUI image widget

Usage::

    pixi run python -m pose_6d.known_board \\
        --video data/insta360_calibration/lens0_combined.mp4 \\
        --intrinsics data/insta360_calibration/lens0_combined_subpixel_best.npz \\
        --marker-config config/apriltag_board.yaml \\
        --grid-name user_10x7_gap30

Streaming: pose computation runs upfront over the whole video (a few minutes
on a 22k-frame source at the default stride). Once it's done, viser playback
is real-time.
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
    apriltag_grid_object_points,
    load_apriltag_grid_configs,
)


@dataclasses.dataclass
class Args:
    """Pose-history viewer for a fisheye calibration recording."""

    video: Path
    """Raw fisheye video (the same input that was fed to calibration)."""

    intrinsics: Path
    """.npz from bench_subpixel.py or pinhole_calibrate.py. Must carry
    ``K``, ``D``, ``image_size``, ``fov_deg``, ``tag_corner_shift``,
    ``pinhole_size``."""

    marker_config: Path = Path("config/apriltag_board.yaml")
    """YAML with the ``apriltag_grid:`` entry to use."""

    grid_name: str = "user_10x7_gap30"
    """Name of the grid inside ``--marker-config``. Defaults to the
    user-confirmed 30mm-gap entry."""

    frame_stride: int = 20
    """Sample every N-th frame. Default 20 -> ~1100 poses on a 22k-frame
    video, which renders smoothly in viser."""

    detect_upscale: int = 8
    """Detection-time bicubic upscale factor. The benchmark found 8x was
    the sweet spot for cv2.aruco's quad detector."""

    coord_frame_stride: int = 30
    """Show a coordinate-frame triad at every Nth precomputed pose. The
    current pose always gets a highlighted triad."""

    axes_length: float = 0.05
    """Coord-frame triad length in metres."""

    axes_radius: float = 0.0025
    """Coord-frame triad axis radius in metres."""

    port: int = 8085


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


def _rvec_tvec_to_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Compose a 4x4 homogeneous transform from solvePnP outputs."""
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec.ravel()
    return T


def _T_to_wxyz_xyz(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a 4x4 transform to viser's (wxyz, xyz) format."""
    R = T[:3, :3]
    t = T[:3, 3]
    # Rotation matrix -> quaternion (w, x, y, z) without scipy.
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
    return np.array([w, x, y, z], dtype=np.float64), t.astype(np.float64)


# ---------------------------------------------------------------------------
# Detection (matches bench_subpixel.py's best method)
# ---------------------------------------------------------------------------


def _detect(bgr: np.ndarray, dict_id: int, allowed_ids: set[int], scale: int):
    h, w = bgr.shape[:2]
    big = cv2.resize(bgr, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(dict_id), params
    )
    rgb = cv2.cvtColor(big, cv2.COLOR_BGR2RGB)
    corners, ids, _ = detector.detectMarkers(rgb)
    if ids is None:
        return None, None
    mask = np.isin(ids.flatten(), list(allowed_ids))
    if not np.any(mask):
        return None, None
    corners = tuple((c.astype(np.float32) / scale) for c, m in zip(corners, mask) if m)
    return corners, ids[mask].reshape(-1, 1)


# ---------------------------------------------------------------------------
# Precompute the full trajectory
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Pose:
    frame_idx: int
    T_board_camera: np.ndarray  # 4x4
    n_tags: int
    cache_idx: int | None = None  # row in the unwrap-cache npy, if available
    corners: np.ndarray | None = None  # (n_tags, 4, 2) detection pixel coords
    ids: np.ndarray | None = None  # (n_tags,) tag IDs


def _find_cache(video: Path, pinhole_size, fov_deg):
    """Locate any unwrap-cache .npy/.meta.json produced by bench_subpixel.py
    for this video + unwrap params. Returns (npy_path, meta) or (None, None)."""
    import json as _json
    parent = video.parent
    for meta_path in parent.glob(f"{video.stem}_unwrap_cache_*.meta.json"):
        with meta_path.open() as f:
            meta = _json.load(f)
        if (tuple(meta["pinhole_size"]) == tuple(pinhole_size)
                and abs(meta["fov_deg"] - fov_deg) < 1e-6):
            npy_path = meta_path.with_suffix("").with_suffix(".npy")
            if npy_path.exists():
                return npy_path, meta
    return None, None


def _find_pose_history(video: Path, intrinsics: Path, grid_name: str) -> Path | None:
    """Look for a precomputed `<stem>_pose_history.npz`. Loaded eagerly to
    skip the detection + PnP pass entirely. Validates the npz references the
    same intrinsics file and grid name; otherwise treats it as stale."""
    candidate = video.parent / f"{video.stem}_pose_history.npz"
    if not candidate.exists():
        return None
    try:
        d = np.load(str(candidate))
        if str(d.get("intrinsics_npz")) not in (intrinsics.name, str(intrinsics)):
            return None
        if str(d.get("grid_name")) != grid_name:
            return None
        return candidate
    except Exception:
        return None


def _precompute_poses(args: Args) -> tuple[list[Pose], tuple[int, int], dict]:
    intr = np.load(str(args.intrinsics))
    K = intr["K"]
    D = intr["D"]
    pinhole_size = tuple(int(v) for v in intr["pinhole_size"])
    fov_deg = float(intr["fov_deg"])
    tag_corner_shift = int(intr["tag_corner_shift"])

    grids = load_apriltag_grid_configs(args.marker_config)
    if args.grid_name not in grids:
        raise RuntimeError(
            f"grid '{args.grid_name}' not in {args.marker_config}; "
            f"available: {list(grids)}"
        )
    g = grids[args.grid_name]
    obj_dict = apriltag_grid_object_points(g)
    allowed_ids = set(g.tag_ids)

    # Fastest path: a previously computed pose-history npz. Skips
    # detection + PnP entirely; loads in milliseconds.
    pose_npz = _find_pose_history(args.video, args.intrinsics, args.grid_name)
    if pose_npz is not None:
        pd = np.load(str(pose_npz), allow_pickle=True)
        Ts = pd["T_board_camera"]
        idxs = pd["frame_indices"]
        n_tags = pd["n_tags"]
        cache_idxs = pd["cache_indices"] if "cache_indices" in pd.files else None
        all_corners = pd["corners"] if "corners" in pd.files else None
        all_ids = pd["ids"] if "ids" in pd.files else None
        # Apply args.frame_stride as a sub-sample selector on the precomputed
        # set, in case the user wants fewer frames.
        sub = max(1, args.frame_stride // 20)  # cache stride was 20
        sel = np.arange(0, len(Ts), sub)
        poses = []
        for i in sel:
            poses.append(Pose(
                frame_idx=int(idxs[i]),
                T_board_camera=Ts[i],
                n_tags=int(n_tags[i]),
                cache_idx=int(cache_idxs[i]) if cache_idxs is not None else None,
                corners=all_corners[i] if all_corners is not None else None,
                ids=all_ids[i] if all_ids is not None else None,
            ))
        print(f"loaded {len(poses)} poses from {pose_npz.name} (precomputed)")
        return poses, pinhole_size, {
            "K": K, "D": D, "pinhole_size": pinhole_size,
            "fov_deg": fov_deg, "tag_corner_shift": tag_corner_shift,
            "board": g, "obj_dict": obj_dict,
        }

    # Cache fast path. The cache was built with a specific frame stride;
    # we treat that stride as the upper sampling resolution and subsample
    # further by args.frame_stride if needed.
    npy_path, meta = _find_cache(args.video, pinhole_size, fov_deg)
    if npy_path is not None:
        cache_stride = meta["frame_stride"]
        sub = max(1, args.frame_stride // cache_stride)
        cached_frames = np.load(npy_path, mmap_mode="r")
        cached_indices = meta["frame_indices"]
        n_total = len(cached_indices)
        print(
            f"using cache {npy_path.name}: {n_total} frames at video-stride "
            f"{cache_stride}; subsampling every {sub} -> {n_total // sub} poses"
        )
        def iter_unwrapped():
            for ci in range(0, n_total, sub):
                yield cached_indices[ci], np.ascontiguousarray(cached_frames[ci])
    else:
        cap = cv2.VideoCapture(str(args.video))
        if not cap.isOpened():
            raise RuntimeError(f"could not open {args.video}")
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"precomputing poses on {args.video} ({n_total} frames, stride={args.frame_stride})")
        rectifier: Rectifier | None = None
        def iter_unwrapped():
            nonlocal rectifier
            idx = -1
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                idx += 1
                if idx % args.frame_stride != 0:
                    continue
                if rectifier is None:
                    fh, fw = bgr.shape[:2]
                    f_eq = fw / math.pi
                    K_fish = np.array(
                        [[f_eq, 0.0, fw / 2.0], [0.0, f_eq, fh / 2.0], [0.0, 0.0, 1.0]],
                        dtype=np.float64,
                    )
                    rectifier = Rectifier.build(K_fish, np.zeros(4), pinhole_size, fov_deg)
                yield idx, rectifier.apply(bgr)
            cap.release()

    poses: list[Pose] = []
    t0 = time.time()
    for idx, unwrapped in iter_unwrapped():
        corners, ids = _detect(unwrapped, g.cv2_dictionary, allowed_ids, args.detect_upscale)
        if ids is None or len(ids) < 4:
            continue
        obj_list, img_list = [], []
        for cc, tid in zip(corners, ids.flatten()):
            tid = int(tid)
            if tid not in obj_dict:
                continue
            op = np.roll(obj_dict[tid], -tag_corner_shift, axis=0)
            obj_list.append(op)
            img_list.append(cc.reshape(4, 2))
        if len(obj_list) < 4:
            continue
        obj_pts = np.vstack(obj_list).astype(np.float32)
        img_pts = np.vstack(img_list).astype(np.float32)
        ok2, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, K, D, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok2:
            continue
        T_camera_board = _rvec_tvec_to_T(rvec, tvec)
        # Invert: camera position in the board frame.
        T_board_camera = np.linalg.inv(T_camera_board)
        poses.append(Pose(idx, T_board_camera, len(obj_list)))
        if len(poses) % 50 == 0:
            print(f"  frame {idx}: {len(poses)} poses recovered")
    dt = time.time() - t0
    print(f"done — {len(poses)} poses in {dt:.1f} s")

    meta = {
        "K": K, "D": D, "pinhole_size": pinhole_size,
        "fov_deg": fov_deg, "tag_corner_shift": tag_corner_shift,
        "board": g, "obj_dict": obj_dict,
    }
    return poses, pinhole_size, meta


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


def _board_outline_segments(g) -> np.ndarray:
    """Sharp-cornered rectangle of the AprilGrid in board frame. Returns an
    (N, 2, 3) array of endpoint pairs suitable for ``add_line_segments``."""
    stride = g.stride
    W = g.tag_cols * stride - (stride - g.tag_size)
    H = g.tag_rows * stride - (stride - g.tag_size)
    pts = np.array(
        [[[0, 0, 0], [W, 0, 0]],
         [[W, 0, 0], [W, H, 0]],
         [[W, H, 0], [0, H, 0]],
         [[0, H, 0], [0, 0, 0]]],
        dtype=np.float32,
    )
    return pts


def _per_tag_marker_pose(K, D, marker_corner_obj, image_corners,
                         T_board_camera: np.ndarray) -> np.ndarray | None:
    """Single-tag PnP -> T_board_marker (marker pose in board frame derived
    from this one observation). Returns 4x4 or None if PnP fails.

    Uses SOLVEPNP_ITERATIVE so we can pair obj_pts and img_pts in our
    own (shift-corrected) order. SOLVEPNP_IPPE_SQUARE would be cheaper but
    requires obj_pts in strict [TL, TR, BR, BL] order, which our 180-deg
    board-shift violates."""
    ok, rvec, tvec = cv2.solvePnP(
        marker_corner_obj.astype(np.float32),
        image_corners.astype(np.float32),
        K, D, flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    T_camera_marker = np.eye(4, dtype=np.float64)
    T_camera_marker[:3, :3] = R
    T_camera_marker[:3, 3] = tvec.ravel()
    # T_board_marker = T_board_camera @ T_camera_marker
    return T_board_camera @ T_camera_marker


def _draw_marker_overlay(bgr: np.ndarray, corners: np.ndarray | None,
                         ids: np.ndarray | None) -> np.ndarray:
    """Return a copy of bgr with green bounding boxes + tag IDs overlaid."""
    out = bgr.copy()
    if corners is None or ids is None or len(corners) == 0:
        return out
    for c, tid in zip(corners, ids):
        pts = c.reshape(-1, 2).astype(np.int32)
        cv2.polylines(out, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        cx, cy = pts.mean(axis=0).astype(int)
        cv2.putText(out, str(int(tid)), (cx - 10, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return out


def main(args: Args) -> None:
    poses, pinhole_size, meta = _precompute_poses(args)
    if not poses:
        raise SystemExit("no poses recovered; check intrinsics/grid config")

    # If the unwrap cache exists and the poses know their cache index,
    # mmap it so the slider can read current frames lazily.
    cache_arr: np.ndarray | None = None
    if poses[0].cache_idx is not None:
        npy_path, _ = _find_cache(args.video, pinhole_size, meta["fov_deg"])
        if npy_path is not None:
            cache_arr = np.load(npy_path, mmap_mode="r")
            print(f"image source: cache {npy_path.name} ({cache_arr.shape[0]} frames)")

    server = viser.ViserServer(port=args.port)

    # --- Static scene -------------------------------------------------------
    g = meta["board"]
    obj_dict = meta["obj_dict"]
    K = meta["K"]
    D = meta["D"]
    tag_corner_shift = meta["tag_corner_shift"]

    # Sharp-cornered board outline as line segments.
    outline_segments = _board_outline_segments(g)
    server.scene.add_line_segments(
        "/board/outline",
        points=outline_segments,
        colors=np.array([220, 220, 220], dtype=np.uint8),
        line_width=2.0,
    )
    # Origin triad on the board.
    server.scene.add_frame(
        "/board/origin", wxyz=(1.0, 0.0, 0.0, 0.0), position=(0, 0, 0),
        axes_length=args.axes_length * 1.5, axes_radius=args.axes_radius * 1.5,
        show_axes=True,
    )

    # --- Static GT marker triads (grey) -------------------------------------
    # Drawn as line segments so we can colour them grey (viser's add_frame
    # always uses red/green/blue axes). One small triad per marker, centred
    # at the marker's GT position on the board, with axes aligned to the
    # marker's physical frame (180-deg rotated vs board due to corner-shift).
    gt_axis_len = g.tag_size * 0.5
    gt_segments = []
    # Per-marker triad: +X (-board_x), +Y (-board_y), +Z (+board_z).
    triad_dirs = np.array(
        [[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32
    ) * gt_axis_len
    for tid, corners_gt in obj_dict.items():
        center = corners_gt.mean(axis=0).astype(np.float32)
        for direction in triad_dirs:
            gt_segments.append([center, center + direction])
    gt_segments = np.asarray(gt_segments, dtype=np.float32)  # (3*N, 2, 3)
    server.scene.add_line_segments(
        "/markers_gt",
        points=gt_segments,
        colors=np.array([180, 180, 180], dtype=np.uint8),
        line_width=1.5,
    )

    # --- Per-marker DYNAMIC coord frames -----------------------------------
    # Pre-allocate one frame handle per possible tag ID. They're hidden until
    # a frame is detected; then we update their pose to T_board_marker
    # computed from this one observation (single-tag PnP), so the user can
    # SEE per-frame marker localisation, not just GT positions.
    s = g.tag_size
    # Marker's own frame: TL/TR/BR/BL of a tag centered at its origin with
    # +X right, +Y up, Z=0. Apply the board's 180-deg corner-shift so the
    # marker frame matches what the detector reports.
    marker_obj_unrot = np.array(
        [[-s/2,  s/2, 0.0],
         [ s/2,  s/2, 0.0],
         [ s/2, -s/2, 0.0],
         [-s/2, -s/2, 0.0]],
        dtype=np.float32,
    )
    marker_obj = np.roll(marker_obj_unrot, -int(tag_corner_shift), axis=0)
    marker_axes_len = s * 0.5
    marker_axes_rad = marker_axes_len * 0.05
    marker_handles: dict[int, viser.FrameHandle] = {}
    for tid in obj_dict.keys():
        marker_handles[int(tid)] = server.scene.add_frame(
            f"/markers/tag_{int(tid):03d}",
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(0.0, 0.0, 0.0),
            axes_length=marker_axes_len,
            axes_radius=marker_axes_rad,
            visible=False,
        )

    # --- Trajectory polyline -----------------------------------------------
    cam_xyz = np.array([p.T_board_camera[:3, 3] for p in poses], dtype=np.float32)
    server.scene.add_spline_catmull_rom(
        "/trajectory",
        positions=cam_xyz,
        color=(80, 180, 255),
        line_width=2.0,
    )

    # --- Sparse history coord frames ---------------------------------------
    for i, p in enumerate(poses):
        if i % args.coord_frame_stride != 0:
            continue
        wxyz, xyz = _T_to_wxyz_xyz(p.T_board_camera)
        server.scene.add_frame(
            f"/history/{i:05d}",
            wxyz=wxyz, position=xyz,
            axes_length=args.axes_length * 0.6,
            axes_radius=args.axes_radius * 0.7,
        )

    # --- Current-pose triad (slider-controlled) -----------------------------
    current_handle = server.scene.add_frame(
        "/current",
        wxyz=(1.0, 0.0, 0.0, 0.0), position=(0, 0, 0),
        axes_length=args.axes_length * 1.6,
        axes_radius=args.axes_radius * 1.6,
        show_axes=True,
    )

    # --- GUI controls ------------------------------------------------------
    info = server.gui.add_text("Poses", initial_value=f"{len(poses)} recovered", disabled=True)
    play_btn = server.gui.add_checkbox("Playing", initial_value=True)
    speed = server.gui.add_slider("Frames/sec", min=1, max=60, step=1, initial_value=20)
    idx_slider = server.gui.add_slider(
        "Pose index", min=0, max=len(poses) - 1, step=1, initial_value=0
    )
    pose_info = server.gui.add_text("Info", initial_value="", disabled=True)
    image_handle = None
    if cache_arr is not None:
        h, w = cache_arr.shape[1], cache_arr.shape[2]
        image_handle = server.gui.add_image(
            np.zeros((h, w, 3), dtype=np.uint8),
            label="Current frame (detections in green)",
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

    cur = 0
    last = time.time()
    while True:
        now = time.time()
        if pending_seek is not None:
            cur = pending_seek
            pending_seek = None
            should_update = True
        elif play_btn.value and (now - last) >= (1.0 / max(speed.value, 1)):
            cur = (cur + 1) % len(poses)
            should_update = True
        else:
            should_update = False

        if should_update:
            p = poses[cur]
            wxyz, xyz = _T_to_wxyz_xyz(p.T_board_camera)
            current_handle.wxyz = tuple(float(v) for v in wxyz)
            current_handle.position = tuple(float(v) for v in xyz)
            pose_info.value = (
                f"frame={p.frame_idx}  tags={p.n_tags}  "
                f"cam_xyz=({xyz[0]:+.3f}, {xyz[1]:+.3f}, {xyz[2]:+.3f}) m"
            )
            if image_handle is not None and cache_arr is not None and p.cache_idx is not None:
                bgr = np.ascontiguousarray(cache_arr[p.cache_idx])
                annotated = _draw_marker_overlay(bgr, p.corners, p.ids)
                # viser's add_image takes RGB
                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                image_handle.image = rgb

            # Update per-marker dynamic coord frames: only the markers DETECTED
            # in this frame are shown; their poses are computed via single-tag
            # PnP from the current observation (not the static GT positions).
            detected_now: set[int] = set()
            if p.corners is not None and p.ids is not None:
                for img_pts, tid in zip(p.corners, p.ids):
                    tid_i = int(tid)
                    if tid_i not in marker_handles:
                        continue
                    T_bm = _per_tag_marker_pose(
                        K, D, marker_obj, np.asarray(img_pts, dtype=np.float32),
                        p.T_board_camera,
                    )
                    if T_bm is None:
                        continue
                    wxyz_m, xyz_m = _T_to_wxyz_xyz(T_bm)
                    h = marker_handles[tid_i]
                    h.wxyz = tuple(float(v) for v in wxyz_m)
                    h.position = tuple(float(v) for v in xyz_m)
                    h.visible = True
                    detected_now.add(tid_i)
            for tid_i, h in marker_handles.items():
                if tid_i not in detected_now:
                    h.visible = False
            muted = True
            idx_slider.value = cur
            muted = False
            last = now
        else:
            time.sleep(0.005)


if __name__ == "__main__":
    main(tyro.cli(Args))
