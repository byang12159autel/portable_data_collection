#!/usr/bin/env python3
"""ArUco-pose -> planar homography pipeline + viser preview.

Given a raw Insta360 ``.insv`` (or a per-lens ``.mp4``), a lens0
calibration ``.npz`` from ``bench_subpixel.py`` / ``pinhole_calibrate.py``,
and a marker YAML, this script:

1. Demuxes the ``.insv`` into a per-lens mp4 if needed
   (``camera.convert``).
2. Equidistant-unwraps each fisheye frame to the calibration's pinhole
   view (Stage 1), then applies ``cv2.undistort`` with the npz's
   ``K, D`` (Stage 2). The resulting frame is a clean pinhole with
   intrinsics ``K`` and zero distortion.
3. Detects the configured ArUco marker, recovers the marker pose with
   ``cv2.solvePnP``, and builds plane <-> image homographies.
4. Overlays the detection, the 3D axes drawn through the homography,
   and a plane-grid; also renders a top-down warp of the marker plane.
5. Writes the annotated frames to an output mp4 and serves a live viser
   preview at ``http://localhost:<port>`` while writing.

Usage::

    pixi run python -m dot_angle_detection.homography_transform \\
        --video data/aruco_test/VID_20260517_192400_00_009.insv \\
        --intrinsics data/insta360_calibration/lens0_combined_subpixel_best.npz \\
        --marker-config config/chopsticks-v1.yaml
"""

from __future__ import annotations

import dataclasses
import math
import sys
import time
from pathlib import Path  # noqa: TC003 — tyro needs Path at runtime

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Pure-math helpers (also imported by callers).
# --------------------------------------------------------------------------


def homography_from_aruco_pose(K, rvec, tvec, z_offset_m: float = 0.0):
    """Build plane-to-image and image-to-plane homographies from an ArUco pose.

    OpenCV ArUco pose convention:
        P_camera = R * P_marker + t

    Target plane (parallel to the marker face, optionally offset along
    the marker's local Z):
        Z_marker = z_offset_m

    Homography:
        s [u, v, 1]^T = K [r1 r2 (z_offset_m * r3 + t)] [X, Y, 1]^T

    A non-zero ``z_offset_m`` lifts (or lowers) the homography to a
    plane parallel to the marker — useful when the features of
    interest (here, the chopstick reference dots) don't lie exactly on
    the marker face. Pass the physical offset in metres; positive is
    the marker's own +Z (out of the marker face), negative is into it.

    Args:
        K:    3x3 camera intrinsic matrix
        rvec: 3x1 rotation vector from ArUco pose estimation
        tvec: 3x1 translation vector from ArUco pose estimation
        z_offset_m: target-plane offset along marker +Z (metres)

    Returns:
        H_plane_to_img: 3x3 homography from target-plane coords (m) to pixels
        H_img_to_plane: 3x3 homography from pixels to target-plane coords (m)
    """

    R, _ = cv2.Rodrigues(rvec)
    t = np.asarray(tvec, dtype=np.float64).reshape(3, 1)

    r1 = R[:, 0:1]
    r2 = R[:, 1:2]
    r3 = R[:, 2:3]

    t_shifted = t + z_offset_m * r3
    H_plane_to_img = K @ np.hstack([r1, r2, t_shifted])
    H_img_to_plane = np.linalg.inv(H_plane_to_img)

    return H_plane_to_img, H_img_to_plane


def apply_homography(H, point_uv):
    """Apply a 3x3 homography to a single 2D point.

    Args:
        H: 3x3 homography
        point_uv: 2D point [u, v]

    Returns:
        Transformed 2D point [x, y].
    """

    u, v = point_uv
    p = np.array([u, v, 1.0], dtype=np.float64)

    q = H @ p
    q = q / q[2]

    return q[:2]


def apply_homography_batch(H, points):
    """Vectorized ``apply_homography`` over an (N, 2) array of points."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    q = (H @ h.T).T
    return q[:, :2] / q[:, 2:3]


# --------------------------------------------------------------------------
# Rectification (Stage 1: equidistant unwrap; Stage 2: pinhole undistort).
# --------------------------------------------------------------------------


def _equidistant_K(src_w: int, src_h: int) -> np.ndarray:
    """Equidistant-fisheye intrinsics: ``f = W/pi``, principal point at center."""
    f_eq = src_w / math.pi
    return np.array(
        [[f_eq, 0.0, src_w / 2.0],
         [0.0, f_eq, src_h / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _pinhole_rough_K(pinhole_size: tuple[int, int], fov_deg: float) -> np.ndarray:
    """Stage-1 unwrap-target intrinsics for the given pinhole size + FOV."""
    fov_rad = math.radians(fov_deg)
    fx = (pinhole_size[0] / 2.0) / math.tan(fov_rad / 2.0)
    return np.array(
        [[fx, 0.0, pinhole_size[0] / 2.0],
         [0.0, fx, pinhole_size[1] / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


@dataclasses.dataclass
class Lens0Rectifier:
    """Composed Stage-1 + Stage-2 remap for the lens0 sub-pixel calibration.

    ``apply(bgr)`` returns a clean pinhole frame with intrinsics ``K``
    and zero distortion. ``K`` is the calibration's refined K from the
    npz, so downstream solvePnP uses ``K`` with ``distCoeffs=0``.
    """

    K: np.ndarray
    D: np.ndarray  # original distortion (kept for diagnostics; PnP uses zeros)
    pinhole_size: tuple[int, int]
    s1_map1: np.ndarray
    s1_map2: np.ndarray
    s2_map1: np.ndarray
    s2_map2: np.ndarray

    @classmethod
    def from_npz(cls, npz_path: Path, src_size: tuple[int, int]) -> "Lens0Rectifier":
        d = np.load(str(npz_path))
        K = np.asarray(d["K"], dtype=np.float64)
        D = np.asarray(d["D"], dtype=np.float64)
        pinhole_size = (int(d["pinhole_size"][0]), int(d["pinhole_size"][1]))
        fov_deg = float(d["fov_deg"])

        # Stage 1: equidistant fisheye -> rough pinhole.
        K_fish = _equidistant_K(src_size[0], src_size[1])
        K_rough = _pinhole_rough_K(pinhole_size, fov_deg)
        s1_map1, s1_map2 = cv2.fisheye.initUndistortRectifyMap(
            K_fish,
            np.zeros(4, dtype=np.float64).reshape(4, 1),
            np.eye(3, dtype=np.float64),
            K_rough,
            pinhole_size,
            cv2.CV_16SC2,
        )
        # Stage 2: pinhole undistort with calibrated K, D back to K (clean).
        s2_map1, s2_map2 = cv2.initUndistortRectifyMap(
            K, D, np.eye(3, dtype=np.float64),
            K, pinhole_size, cv2.CV_16SC2,
        )
        return cls(
            K=K, D=D, pinhole_size=pinhole_size,
            s1_map1=s1_map1, s1_map2=s1_map2,
            s2_map1=s2_map1, s2_map2=s2_map2,
        )

    def apply(self, bgr: np.ndarray) -> np.ndarray:
        unwrapped = cv2.remap(bgr, self.s1_map1, self.s1_map2, cv2.INTER_LINEAR)
        return cv2.remap(unwrapped, self.s2_map1, self.s2_map2, cv2.INTER_LINEAR)


# --------------------------------------------------------------------------
# Detection + PnP.
# --------------------------------------------------------------------------


def _marker_object_points(size: float) -> np.ndarray:
    """4 corners of a planar square marker at z=0, ordered [TL, TR, BR, BL]."""
    half = size / 2.0
    return np.array(
        [[-half,  half, 0.0],
         [ half,  half, 0.0],
         [ half, -half, 0.0],
         [-half, -half, 0.0]],
        dtype=np.float32,
    )


def _detect_marker(frame_bgr: np.ndarray, dict_id: int, target_id: int):
    """Detect ``target_id`` in ``frame_bgr``; return its 4 image corners or None."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, _ = detector.detectMarkers(frame_bgr)
    if ids is None:
        return None
    flat = ids.flatten()
    matches = np.where(flat == target_id)[0]
    if len(matches) == 0:
        return None
    return corners[int(matches[0])].reshape(4, 2).astype(np.float32)


# --------------------------------------------------------------------------
# Drawing helpers.
# --------------------------------------------------------------------------


def _draw_detection(img: np.ndarray, corners: np.ndarray, tag_id: int) -> None:
    pts = corners.astype(np.int32)
    cv2.polylines(img, [pts], True, (0, 255, 0), 2)
    # Corner labels (TL, TR, BR, BL).
    for label, (x, y) in zip(("TL", "TR", "BR", "BL"), pts):
        cv2.circle(img, (int(x), int(y)), 4, (0, 255, 255), -1)
        cv2.putText(img, label, (int(x) + 6, int(y) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    cx, cy = pts.mean(axis=0).astype(int)
    cv2.putText(img, f"id={tag_id}", (cx - 20, cy + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)


def _draw_axes_via_homography(img: np.ndarray, H_plane_to_img: np.ndarray,
                              length_m: float) -> None:
    """Draw the marker's XYZ axes by warping plane points through H.

    Note: Z is approximated as a small out-of-plane segment via the marker
    pose; for the in-plane axes we rely solely on the homography (no
    cv2.projectPoints needed).
    """
    origin = apply_homography(H_plane_to_img, (0.0, 0.0))
    x_end = apply_homography(H_plane_to_img, (length_m, 0.0))
    y_end = apply_homography(H_plane_to_img, (0.0, length_m))
    o = tuple(int(v) for v in origin)
    cv2.arrowedLine(img, o, tuple(int(v) for v in x_end), (0, 0, 255), 3,
                    tipLength=0.15)  # X red
    cv2.arrowedLine(img, o, tuple(int(v) for v in y_end), (0, 255, 0), 3,
                    tipLength=0.15)  # Y green


def _draw_z_axis(img: np.ndarray, K: np.ndarray, rvec: np.ndarray,
                 tvec: np.ndarray, length_m: float) -> None:
    """Z axis needs the full 3D projection; draw it on top of the homography axes."""
    pts_3d = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, length_m]], dtype=np.float32
    )
    pts_2d, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, np.zeros(5))
    p0 = tuple(int(v) for v in pts_2d[0, 0])
    p1 = tuple(int(v) for v in pts_2d[1, 0])
    cv2.arrowedLine(img, p0, p1, (255, 0, 0), 3, tipLength=0.15)  # Z blue


def _draw_plane_grid(img: np.ndarray, H_plane_to_img: np.ndarray,
                     extent_m: float, step_m: float) -> None:
    """Draw a square grid on the marker plane via the homography."""
    h, w = img.shape[:2]
    ticks = np.arange(-extent_m, extent_m + step_m * 0.5, step_m)
    # Horizontal lines (constant Y).
    for y in ticks:
        pts = apply_homography_batch(
            H_plane_to_img, np.column_stack([ticks, np.full_like(ticks, y)])
        ).astype(np.int32)
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            if 0 <= x0 < w and 0 <= y0 < h and 0 <= x1 < w and 0 <= y1 < h:
                cv2.line(img, (x0, y0), (x1, y1), (255, 200, 80), 1, cv2.LINE_AA)
    # Vertical lines (constant X).
    for x in ticks:
        pts = apply_homography_batch(
            H_plane_to_img, np.column_stack([np.full_like(ticks, x), ticks])
        ).astype(np.int32)
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            if 0 <= x0 < w and 0 <= y0 < h and 0 <= x1 < w and 0 <= y1 < h:
                cv2.line(img, (x0, y0), (x1, y1), (255, 200, 80), 1, cv2.LINE_AA)


def _birdseye(img: np.ndarray, H_plane_to_img: np.ndarray,
              extent_m: float, width_px: int,
              height_px: int | None = None) -> np.ndarray:
    """Warp the image into a top-down view of the marker plane.

    The plane origin maps to the center of the output, +X right, +Y up.
    ``extent_m`` is the *x* half-extent; the y half-extent is derived
    from ``height_px`` so the pixels-per-meter scale matches on both
    axes (no aspect-ratio distortion).
    """
    if height_px is None:
        height_px = width_px
    s = width_px / (2.0 * extent_m)            # pixels per metre
    extent_y_m = height_px / (2.0 * s)         # half-extent on Y
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
    cx, cy = width_px // 2, height_px // 2
    cv2.line(out, (0, cy), (width_px, cy), (255, 200, 80), 1)
    cv2.line(out, (cx, 0), (cx, height_px), (255, 200, 80), 1)
    cv2.arrowedLine(out, (cx, cy), (cx + int(s * extent_m * 0.5), cy),
                    (0, 0, 255), 2, tipLength=0.15)
    cv2.arrowedLine(out, (cx, cy), (cx, cy - int(s * extent_y_m * 0.5)),
                    (0, 255, 0), 2, tipLength=0.15)
    cv2.putText(out,
                f"birds-eye  x+-{extent_m * 1000:.0f}mm  y+-{extent_y_m * 1000:.0f}mm",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _embed_inset(canvas: np.ndarray, inset: np.ndarray, margin: int = 12) -> None:
    h, w = canvas.shape[:2]
    ih, iw = inset.shape[:2]
    y0 = h - ih - margin
    x0 = w - iw - margin
    canvas[y0:y0 + ih, x0:x0 + iw] = inset
    cv2.rectangle(canvas, (x0 - 1, y0 - 1), (x0 + iw, y0 + ih),
                  (255, 255, 255), 1)


# --------------------------------------------------------------------------
# Top-level pipeline.
# --------------------------------------------------------------------------


@dataclasses.dataclass
class Args:
    """ArUco-pose homography pipeline."""

    video: Path
    """Source recording — accepts .insv (auto-demuxes lens0) or .mp4."""

    intrinsics: Path
    """Per-lens0 calibration .npz (must carry K, D, pinhole_size, fov_deg)."""

    marker_config: Path
    """YAML with a ``markers:`` section. Detects the first entry by default;
    override with ``--marker-name``."""

    marker_name: str | None = None
    """Name of the marker entry inside ``--marker-config``. Default: first."""

    output: Path | None = None
    """Output mp4 path. Defaults to ``<video stem>_homography.mp4``
    alongside the source."""

    port: int = 8085
    """Viser preview port."""

    fps: float = 0.0
    """Override playback FPS. 0 = use source FPS."""

    serve_viser: bool = True
    """Stream a live viser preview while processing."""

    axes_length_factor: float = 1.5
    """3D axes length as a multiple of marker size."""

    grid_extent_factor: float = 3.0
    """Plane grid half-extent as a multiple of marker size."""

    grid_step_factor: float = 0.5
    """Plane grid step as a multiple of marker size."""

    birdseye_extent_factor: float = 4.0
    """Bird's-eye view half-extent as a multiple of marker size."""

    birdseye_px: int = 320
    """Bird's-eye inset width in pixels."""

    birdseye_height_px: int | None = 640
    """Bird's-eye inset height in pixels. Set to ``None`` (or match
    ``birdseye_px``) for a square inset. Pixels-per-meter scale stays
    identical on both axes; height just shows more of the Y plane."""

    force_demux: bool = False
    """If --video is .insv, overwrite any existing lens0 mp4."""


def _resolve_lens0_mp4(video: Path, force: bool) -> Path:
    if video.suffix.lower() != ".insv":
        return video
    lens0 = video.with_name(video.stem + "_lens0.mp4")
    if lens0.exists() and not force:
        print(f"reusing existing {lens0}")
        return lens0
    # Defer the import so this module stays importable when pose_calibration
    # isn't on sys.path (tests, downstream consumers).
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pose_calibration.camera.convert import convert
    lens1 = video.with_name(video.stem + "_lens1.mp4")
    print(f"demuxing {video} -> {lens0.name}, {lens1.name}")
    convert(video, lens0, lens1, force=force)
    return lens0


def _load_marker(marker_config: Path, marker_name: str | None):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pose_calibration.markers.detect import (
        load_marker_configs, resolve_aruco_dict,
    )
    markers = load_marker_configs(marker_config)
    if not markers:
        raise SystemExit(f"no markers defined in {marker_config}")
    if marker_name is None:
        marker_name, marker_cfg = next(iter(markers.items()))
    else:
        if marker_name not in markers:
            raise SystemExit(
                f"marker '{marker_name}' not in {marker_config}; "
                f"available: {list(markers)}"
            )
        marker_cfg = markers[marker_name]
    dict_id = resolve_aruco_dict(marker_cfg.dictionary)
    print(
        f"marker '{marker_name}': id={marker_cfg.id} size={marker_cfg.size} m "
        f"dict={marker_cfg.dictionary}"
    )
    return marker_cfg, dict_id


def main(args: Args) -> None:
    lens0_mp4 = _resolve_lens0_mp4(args.video, args.force_demux)
    marker_cfg, dict_id = _load_marker(args.marker_config, args.marker_name)

    cap = cv2.VideoCapture(str(lens0_mp4))
    if not cap.isOpened():
        raise SystemExit(f"could not open {lens0_mp4}")
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"video: {lens0_mp4} {src_w}x{src_h} @ {src_fps:.1f} FPS, {n_frames} frames")

    rect = Lens0Rectifier.from_npz(args.intrinsics, (src_w, src_h))
    out_w, out_h = rect.pinhole_size
    print(f"undistort -> {out_w}x{out_h} pinhole (K fx={rect.K[0,0]:.1f})")

    out_path = args.output or args.video.with_name(args.video.stem + "_homography.mp4")
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
    if args.serve_viser:
        try:
            import viser
            server = viser.ViserServer(port=args.port)
            image_handle = server.gui.add_image(
                np.zeros((out_h, out_w, 3), dtype=np.uint8),
                label="lens0 + homography overlay",
            )
            status_handle = server.gui.add_text(
                "Status", initial_value="starting...", disabled=True,
            )
            print(f"viser preview: http://localhost:{args.port}")
        except Exception as e:  # pragma: no cover
            print(f"(viser disabled: {e})")
            server = None

    K = rect.K
    distC = np.zeros(5, dtype=np.float64)
    obj_pts = _marker_object_points(marker_cfg.size)
    axes_len = marker_cfg.size * args.axes_length_factor
    grid_extent = marker_cfg.size * args.grid_extent_factor
    grid_step = marker_cfg.size * args.grid_step_factor
    birds_extent = marker_cfg.size * args.birdseye_extent_factor

    n_done = 0
    n_pose = 0
    t0 = time.time()
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            undistorted = rect.apply(bgr)
            canvas = undistorted.copy()

            corners = _detect_marker(undistorted, dict_id, marker_cfg.id)
            pose_ok = False
            if corners is not None:
                ok_pnp, rvec, tvec = cv2.solvePnP(
                    obj_pts, corners, K, distC,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE,
                )
                if ok_pnp:
                    pose_ok = True
                    H_plane_to_img, _ = homography_from_aruco_pose(K, rvec, tvec)
                    _draw_plane_grid(canvas, H_plane_to_img, grid_extent, grid_step)
                    _draw_detection(canvas, corners, marker_cfg.id)
                    _draw_axes_via_homography(canvas, H_plane_to_img, axes_len)
                    _draw_z_axis(canvas, K, rvec, tvec, axes_len)
                    inset = _birdseye(
                        undistorted, H_plane_to_img, birds_extent,
                        args.birdseye_px, args.birdseye_height_px,
                    )
                    _embed_inset(canvas, inset)
                    dist = float(np.linalg.norm(tvec))
                    cv2.putText(
                        canvas, f"frame {n_done}  d={dist*1000:.1f} mm",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2, cv2.LINE_AA,
                    )
                    n_pose += 1

            if not pose_ok:
                cv2.putText(
                    canvas, f"frame {n_done}  (no marker)",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (60, 60, 255), 2, cv2.LINE_AA,
                )

            writer.write(canvas)
            n_done += 1
            if image_handle is not None and n_done % 2 == 0:
                image_handle.image = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
                if status_handle is not None:
                    fps_now = n_done / max(time.time() - t0, 1e-3)
                    status_handle.value = (
                        f"frame {n_done}/{n_frames}  poses={n_pose}  "
                        f"{fps_now:.1f} fps"
                    )
            if n_done % 60 == 0:
                fps_now = n_done / max(time.time() - t0, 1e-3)
                print(
                    f"  {n_done}/{n_frames}  poses={n_pose}  {fps_now:.1f} fps"
                )
    finally:
        cap.release()
        writer.release()

    print(f"wrote {out_path} ({n_done} frames, {n_pose} with pose)")
    if server is not None:
        if status_handle is not None:
            status_handle.value = (
                f"done — {n_done} frames, {n_pose} with pose. "
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
