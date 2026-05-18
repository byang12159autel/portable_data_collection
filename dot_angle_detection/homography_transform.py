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
import time
from pathlib import Path  # noqa: TC003 — tyro needs Path at runtime

import cv2
import numpy as np

from core.camera.convert import convert as _insv_demux
from core.geometry import homography_from_aruco_pose
from core.markers import load_marker_configs, resolve_aruco_dict
from core.viz.birdseye import birdseye, embed_inset
from core.viz.overlays import (
    draw_axes_via_homography,
    draw_detection,
    draw_plane_grid,
    draw_z_axis,
)


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
    lens1 = video.with_name(video.stem + "_lens1.mp4")
    print(f"demuxing {video} -> {lens0.name}, {lens1.name}")
    _insv_demux(video, lens0, lens1, force=force)
    return lens0


def _load_marker(marker_config: Path, marker_name: str | None):
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
                    draw_plane_grid(canvas, H_plane_to_img, grid_extent, grid_step)
                    draw_detection(canvas, corners, marker_cfg.id)
                    draw_axes_via_homography(canvas, H_plane_to_img, axes_len)
                    draw_z_axis(canvas, K, rvec, tvec, axes_len)
                    inset = birdseye(
                        undistorted, H_plane_to_img, birds_extent,
                        args.birdseye_px, args.birdseye_height_px,
                    )
                    embed_inset(canvas, inset)
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
