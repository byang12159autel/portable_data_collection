#!/usr/bin/env python3
"""Benchmark sub-pixel corner-refinement methods against the Insta360
'single lens unwarped' calibration video.

Each invocation:
  1. opens the video
  2. detects + refines AprilGrid corners with a named method
  3. runs cv2.calibrateCamera with fix-aspect-ratio + 5-coef pinhole
  4. appends a JSONL line to bench_subpixel.jsonl

Add new methods by appending a function to METHODS at the bottom of this
file. The signature is::

    method(bgr: np.ndarray, dict_id: int, allowed_ids: set[int]
           ) -> tuple[tuple[np.ndarray, ...] | None, np.ndarray | None]

i.e. mirror cv2.aruco.ArucoDetector.detectMarkers's return, but you may
do anything you want internally (preprocess, alternative detectors,
manual cv2.cornerSubPix passes, etc.).

Usage::

    pixi run python bench_subpixel.py --method baseline_aruco_subpix
    pixi run python bench_subpixel.py --list
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np

from pose_calibration.calibration.rectify import Rectifier
from pose_calibration.markers.detect import (
    apriltag_grid_object_points,
    load_apriltag_grid_configs,
)

VIDEO = Path("data/insta360_calibration/lens0_combined.mp4")
MARKER_CONFIG = Path("config/apriltag_board.yaml")
GRID_NAME = "user_10x7_gap30"  # user-confirmed 30mm gap, see YAML note
RESULTS_LOG = Path("bench_subpixel.jsonl")
# The physical board's tags are mounted 180-deg from Kalibr convention;
# checked empirically on both this video and the app-unwarp video.
TAG_CORNER_SHIFT = 2
# Equidistant Stage-1 unwrap params (match two_stage_calibrate.py defaults).
PINHOLE_SIZE = (1280, 1280)
FOV_DEG = 110.0

DetectFn = Callable[[np.ndarray, int, set[int]], tuple[tuple | None, np.ndarray | None]]


def _filter_ids(corners: tuple, ids: np.ndarray | None, allowed_ids: set[int]):
    if ids is None:
        return None, None
    mask = np.isin(ids.flatten(), list(allowed_ids))
    if not np.any(mask):
        return None, None
    return tuple(c for c, m in zip(corners, mask) if m), ids[mask].reshape(-1, 1)


# ---------------------------------------------------------------------------
# Detection methods
# ---------------------------------------------------------------------------


def m_baseline_aruco_subpix(bgr, dict_id, allowed_ids):
    """Current baseline: aruco with CORNER_REFINE_SUBPIX (~8.3 px RMS)."""
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(dict_id), params)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    corners, ids, _ = detector.detectMarkers(rgb)
    return _filter_ids(corners, ids, allowed_ids)


def m_aruco_apriltag_refine(bgr, dict_id, allowed_ids):
    """Aruco with CORNER_REFINE_APRILTAG. NOTE: this mode returns 0 markers
    on OpenCV 4.10 in this environment (verified empirically). Left here for
    completeness so other versions can re-run it. See OpenCV issues #2643
    (aruco division-by-zero) and #23437 (CONTOUR regression in 4.7+)."""
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(dict_id), params)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    corners, ids, _ = detector.detectMarkers(rgb)
    return _filter_ids(corners, ids, allowed_ids)


# Lazy-initialised, shared across calls so we don't rebuild the detector
# per frame. pupil-apriltags is C++-backed and stateless once constructed.
_PUPIL_DETECTOR = None


def _pupil_detector():
    global _PUPIL_DETECTOR
    if _PUPIL_DETECTOR is None:
        import pupil_apriltags
        _PUPIL_DETECTOR = pupil_apriltags.Detector(
            families="tag36h11",
            nthreads=1,  # the outer loop is already CPU-saturating
            quad_decimate=1.0,  # don't decimate; we want max accuracy
            refine_edges=1,  # default; documented to improve sub-pixel
        )
    return _PUPIL_DETECTOR


def _pupil_to_aruco_corners(detections, allowed_ids):
    """Convert pupil-apriltags Detection objects to the (corners, ids) tuple
    that the harness expects. pupil-apriltags returns corners as
    [bottom-left, bottom-right, top-right, top-left] in image coordinates
    (note: y-axis flipped vs OpenCV). OpenCV aruco's order is
    [TL, TR, BR, BL]. So we reverse pupil's order and roll by 1 to align."""
    if not detections:
        return None, None
    corners = []
    ids = []
    for det in detections:
        if det.tag_id not in allowed_ids:
            continue
        # pupil: [BL, BR, TR, TL] (image-space; image y points down)
        # OpenCV aruco TL/TR/BR/BL in IMAGE COORDS (y-down):
        #   image-TL = upper-left of upright tag = pupil's TL (last)
        # So map pupil [BL, BR, TR, TL] -> aruco [TL, TR, BR, BL]
        # i.e. cyclic reverse: [TL, BL, BR, TR] then permute... easier:
        # aruco order [TL, TR, BR, BL] = pupil [3, 2, 1, 0] = reversed
        c = np.asarray(det.corners, dtype=np.float32)[::-1].copy()  # (4, 2)
        corners.append(c.reshape(1, 4, 2))
        ids.append(det.tag_id)
    if not corners:
        return None, None
    return tuple(corners), np.array(ids, dtype=np.int32).reshape(-1, 1)


def m_pupil_apriltags_default(bgr, dict_id, allowed_ids):
    """pupil-apriltags (official AprilTag 3 detector via Python bindings).
    Documented to give 0.1-0.3 px corner accuracy on good imagery — much
    better than cv2.aruco's SUBPIX refinement. Refs:
    https://github.com/pupil-labs/apriltags
    Wang & Olson 2016 (AprilTag 3 paper).

    NOTE: empirically detects FEWER tags than aruco on this dataset — close-up
    frames with the most tags return 0 detections from pupil. Keep for ref."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    detections = _pupil_detector().detect(gray)
    return _pupil_to_aruco_corners(detections, allowed_ids)


def m_aruco_norefine(bgr, dict_id, allowed_ids):
    """Lower-bound: no sub-pixel refinement at all. Quantifies how much
    SUBPIX is actually helping vs the underlying quad detector."""
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(dict_id), params)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    corners, ids, _ = detector.detectMarkers(rgb)
    return _filter_ids(corners, ids, allowed_ids)


def m_aruco_contour(bgr, dict_id, allowed_ids):
    """CORNER_REFINE_CONTOUR — fits a polygon to the marker contour and
    extracts corners. (Note: opencv/opencv#23437 reports this had a regression
    in 4.7+. Still worth a benchmark to see if 4.10 is fixed.)"""
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(dict_id), params)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    corners, ids, _ = detector.detectMarkers(rgb)
    return _filter_ids(corners, ids, allowed_ids)


def _aruco_subpix(bgr, dict_id, allowed_ids, *, win_size=5, max_iter=30, min_acc=0.1):
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = win_size
    params.cornerRefinementMaxIterations = max_iter
    params.cornerRefinementMinAccuracy = min_acc
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(dict_id), params)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    corners, ids, _ = detector.detectMarkers(rgb)
    return _filter_ids(corners, ids, allowed_ids)


def m_aruco_subpix_w3(bgr, dict_id, allowed_ids):
    """Sub-pixel refinement window 3 (smaller than default 5)."""
    return _aruco_subpix(bgr, dict_id, allowed_ids, win_size=3)


def m_aruco_subpix_w7(bgr, dict_id, allowed_ids):
    return _aruco_subpix(bgr, dict_id, allowed_ids, win_size=7)


def m_aruco_subpix_w11(bgr, dict_id, allowed_ids):
    return _aruco_subpix(bgr, dict_id, allowed_ids, win_size=11)


def m_aruco_subpix_w7_tight(bgr, dict_id, allowed_ids):
    """Window 7, more iterations, tighter accuracy threshold."""
    return _aruco_subpix(bgr, dict_id, allowed_ids,
                         win_size=7, max_iter=100, min_acc=0.001)


def m_clahe_then_subpix(bgr, dict_id, allowed_ids):
    """CLAHE-enhance gray channel, then standard SUBPIX. CLAHE local-contrast
    normalisation often helps when board lighting is uneven."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    bgr2 = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return m_baseline_aruco_subpix(bgr2, dict_id, allowed_ids)


def _upscale_detect_downscale(bgr, dict_id, allowed_ids, scale=2, refine=cv2.aruco.CORNER_REFINE_SUBPIX):
    """Resize image up, detect on upscaled (more sub-pixel resolution),
    divide corner coordinates by `scale` to bring back to original frame."""
    h, w = bgr.shape[:2]
    big = cv2.resize(bgr, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = refine
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(dict_id), params)
    rgb = cv2.cvtColor(big, cv2.COLOR_BGR2RGB)
    corners, ids, _ = detector.detectMarkers(rgb)
    if ids is None:
        return None, None
    # Scale corners back to original-frame coordinates.
    corners_scaled = tuple((c.astype(np.float32) / scale) for c in corners)
    return _filter_ids(corners_scaled, ids, allowed_ids)


def m_upscale2x_subpix(bgr, dict_id, allowed_ids):
    """Detect on 2x bicubic-upscaled image, scale corners back. The detector
    sees twice the pixel grid for sub-pixel localisation."""
    return _upscale_detect_downscale(bgr, dict_id, allowed_ids, scale=2,
                                     refine=cv2.aruco.CORNER_REFINE_SUBPIX)


def m_upscale2x_norefine(bgr, dict_id, allowed_ids):
    """2x upscale + NO refinement — relies purely on the quad detector at
    higher resolution."""
    return _upscale_detect_downscale(bgr, dict_id, allowed_ids, scale=2,
                                     refine=cv2.aruco.CORNER_REFINE_NONE)


def m_upscale4x_norefine(bgr, dict_id, allowed_ids):
    """4x upscale (memory: 1280*4 = 5120; 75 MB per gray frame)."""
    return _upscale_detect_downscale(bgr, dict_id, allowed_ids, scale=4,
                                     refine=cv2.aruco.CORNER_REFINE_NONE)


def m_upscale8x_norefine(bgr, dict_id, allowed_ids):
    """8x upscale — pushes the 4x trend (1280*8 = 10240; ~100 MB / gray frame)."""
    return _upscale_detect_downscale(bgr, dict_id, allowed_ids, scale=8,
                                     refine=cv2.aruco.CORNER_REFINE_NONE)


def m_upscale16x_norefine(bgr, dict_id, allowed_ids):
    """16x upscale — extreme; 1280*16 = 20480 (~1.3 GB grayscale). May OOM
    or hit OpenCV size limits; benchmark anyway to see diminishing-return point."""
    return _upscale_detect_downscale(bgr, dict_id, allowed_ids, scale=16,
                                     refine=cv2.aruco.CORNER_REFINE_NONE)


# ---------------------------------------------------------------------------
# Calibration harness
# ---------------------------------------------------------------------------


def _cache_paths(
    pinhole_size: tuple[int, int], fov_deg: float, frame_stride: int
) -> tuple[Path, Path]:
    """One cache file per (video, pinhole_size, fov_deg, frame_stride)."""
    cache_dir = VIDEO.parent
    stem = (
        f"{VIDEO.stem}_unwrap_cache_"
        f"{pinhole_size[0]}x{pinhole_size[1]}_fov{fov_deg:.0f}_str{frame_stride}"
    )
    return cache_dir / f"{stem}.npy", cache_dir / f"{stem}.meta.json"


def build_unwrap_cache(
    pinhole_size: tuple[int, int] = PINHOLE_SIZE,
    fov_deg: float = FOV_DEG,
    frame_stride: int = 20,
    overwrite: bool = False,
) -> Path:
    """One-time prepass: decode + unwrap every Nth video frame into a single
    .npy file on disk, plus a sibling .meta.json. Subsequent benchmark runs
    memory-map this file instead of re-decoding the H.265 source.

    Returns the .npy path.
    """
    npy_path, meta_path = _cache_paths(pinhole_size, fov_deg, frame_stride)
    if npy_path.exists() and meta_path.exists() and not overwrite:
        with meta_path.open() as f:
            meta = json.load(f)
        print(f"cache already present: {npy_path} (n_frames={meta['n_frames']})")
        return npy_path

    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {VIDEO}")
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(
        f"building cache from {VIDEO} ({n_total} frames, stride={frame_stride}) "
        f"-> {npy_path}"
    )

    rectifier: Rectifier | None = None
    selected: list[int] = []
    n_kept = (n_total + frame_stride - 1) // frame_stride
    pH, pW = pinhole_size[1], pinhole_size[0]
    # Pre-allocate the npy on disk (uncompressed; uint8 H W 3 per frame).
    # Use np.lib.format.open_memmap so we can write frames as we go.
    arr = np.lib.format.open_memmap(
        npy_path, mode="w+", dtype=np.uint8, shape=(n_kept, pH, pW, 3)
    )
    write_idx = 0

    t0 = time.time()
    idx = -1
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        idx += 1
        if idx % frame_stride != 0:
            continue
        if rectifier is None:
            fh, fw = bgr.shape[:2]
            f_eq = fw / math.pi
            K_fish = np.array(
                [[f_eq, 0.0, fw / 2.0], [0.0, f_eq, fh / 2.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            rectifier = Rectifier.build(K_fish, np.zeros(4), pinhole_size, fov_deg)
        unwrapped = rectifier.apply(bgr)
        arr[write_idx] = unwrapped
        selected.append(idx)
        write_idx += 1
        if write_idx % 100 == 0:
            print(f"  {idx}/{n_total}: cached {write_idx} frames")
    cap.release()
    # Truncate any pre-allocated tail (in case real count < n_kept).
    if write_idx < n_kept:
        # Reopen with the actual size.
        del arr
        full = np.load(npy_path, mmap_mode="r")
        np.save(npy_path, full[:write_idx])
    else:
        del arr  # flush memmap to disk

    meta = {
        "video": str(VIDEO),
        "pinhole_size": list(pinhole_size),
        "fov_deg": fov_deg,
        "frame_stride": frame_stride,
        "n_frames": write_idx,
        "frame_indices": selected,
    }
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)

    dt = time.time() - t0
    size_mb = npy_path.stat().st_size / (1024 * 1024)
    print(f"cache built: {write_idx} frames, {size_mb:.1f} MB, {dt:.1f} s")
    return npy_path


def _iter_unwrapped_frames(
    pinhole_size: tuple[int, int], fov_deg: float, frame_stride: int,
    *, use_cache: bool = True,
) -> Iterator[np.ndarray]:
    """Yield unwrapped BGR frames. Memory-maps the cache if it exists,
    otherwise decodes + unwraps live."""
    npy_path, meta_path = _cache_paths(pinhole_size, fov_deg, frame_stride)
    if use_cache and npy_path.exists() and meta_path.exists():
        arr = np.load(npy_path, mmap_mode="r")
        print(f"  reading cache {npy_path.name} ({arr.shape[0]} frames)")
        for i in range(arr.shape[0]):
            yield np.ascontiguousarray(arr[i])
        return

    # Fall-through: decode + unwrap live (same as before).
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {VIDEO}")
    rectifier: Rectifier | None = None
    idx = -1
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        idx += 1
        if idx % frame_stride != 0:
            continue
        if rectifier is None:
            fh, fw = bgr.shape[:2]
            f_eq = fw / math.pi
            K_fish = np.array(
                [[f_eq, 0.0, fw / 2.0], [0.0, f_eq, fh / 2.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            rectifier = Rectifier.build(K_fish, np.zeros(4), pinhole_size, fov_deg)
        yield rectifier.apply(bgr)
    cap.release()


def run_benchmark(
    method_name: str,
    method_fn: DetectFn,
    method_params: dict | None = None,
    frame_stride: int = 20,
    min_corners_per_view: int = 8,
    min_views: int = 10,
    fix_aspect_ratio: bool = True,
    rational_model: bool = False,
    pinhole_size: tuple[int, int] = PINHOLE_SIZE,
    fov_deg: float = FOV_DEG,
    use_cache: bool = True,
    outlier_filter_pct: float = 0.0,
):
    grids = load_apriltag_grid_configs(MARKER_CONFIG)
    if GRID_NAME not in grids:
        raise RuntimeError(f"grid '{GRID_NAME}' not in {MARKER_CONFIG}; available: {list(grids)}")
    grid_cfg = grids[GRID_NAME]
    dict_id = grid_cfg.cv2_dictionary
    allowed_ids = set(grid_cfg.tag_ids)
    grid_obj_pts = apriltag_grid_object_points(grid_cfg)

    obj_views: list[np.ndarray] = []
    img_views: list[np.ndarray] = []
    image_size: tuple[int, int] = pinhole_size

    t0 = time.time()
    for bgr in _iter_unwrapped_frames(pinhole_size, fov_deg, frame_stride,
                                      use_cache=use_cache):
        corners, ids = method_fn(bgr, dict_id, allowed_ids)
        if ids is None:
            continue
        obj_list, img_list = [], []
        for c, tid in zip(corners, ids.flatten()):
            tid = int(tid)
            if tid not in grid_obj_pts:
                continue
            op = np.roll(grid_obj_pts[tid], -TAG_CORNER_SHIFT, axis=0)
            obj_list.append(op)
            img_list.append(c.reshape(4, 2))
        if not obj_list or sum(len(p) for p in obj_list) < min_corners_per_view:
            continue
        obj_views.append(np.vstack(obj_list).astype(np.float32).reshape(-1, 1, 3))
        img_views.append(np.vstack(img_list).astype(np.float32).reshape(-1, 1, 2))
    t_detect = time.time() - t0

    if len(obj_views) < min_views:
        result = {
            "method": method_name,
            "params": method_params or {},
            "status": "too_few_views",
            "n_views": len(obj_views),
            "t_detect_s": round(t_detect, 1),
        }
        _append_result(result)
        return result

    w, h = image_size
    f_init = 0.6 * w
    K_init = np.array([[f_init, 0, w / 2], [0, f_init, h / 2], [0, 0, 1]], dtype=np.float64)

    flags = cv2.CALIB_USE_INTRINSIC_GUESS
    if fix_aspect_ratio:
        flags |= cv2.CALIB_FIX_ASPECT_RATIO
    if rational_model:
        flags |= cv2.CALIB_RATIONAL_MODEL
    else:
        flags |= cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3

    t1 = time.time()
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
        obj_views, img_views, image_size, K_init, None, flags=flags
    )
    t_calibrate = time.time() - t1

    # Optional outlier filter: drop the top X% of views by per-view RMS and
    # re-calibrate. The per-view RMS for the filter pass is computed against
    # the first-pass K/D.
    dropped_views = 0
    if outlier_filter_pct > 0.0:
        per_view_rms_first: list[float] = []
        for obj, img, r, t in zip(obj_views, img_views, rvecs, tvecs):
            rep, _ = cv2.projectPoints(obj, r, t, K, D)
            e = np.linalg.norm(rep.reshape(-1, 2) - img.reshape(-1, 2), axis=1)
            per_view_rms_first.append(float(math.sqrt((e ** 2).mean())))
        cutoff = float(np.percentile(per_view_rms_first, 100.0 - outlier_filter_pct))
        keep = [v for v, prv in zip(zip(obj_views, img_views), per_view_rms_first)
                if prv <= cutoff]
        dropped_views = len(obj_views) - len(keep)
        obj_views = [v[0] for v in keep]
        img_views = [v[1] for v in keep]
        if len(obj_views) >= min_views:
            t1b = time.time()
            rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
                obj_views, img_views, image_size, K_init, None, flags=flags
            )
            t_calibrate += time.time() - t1b

    # Per-radius residual histogram + per-view RMS.
    cx, cy = float(K[0, 2]), float(K[1, 2])
    per_view_rms: list[float] = []
    all_errs: list[np.ndarray] = []
    all_radii: list[np.ndarray] = []
    for obj, img, r, t in zip(obj_views, img_views, rvecs, tvecs):
        rep, _ = cv2.projectPoints(obj, r, t, K, D)
        e = np.linalg.norm(rep.reshape(-1, 2) - img.reshape(-1, 2), axis=1)
        rad = np.linalg.norm(img.reshape(-1, 2) - [cx, cy], axis=1)
        per_view_rms.append(float(math.sqrt((e ** 2).mean())))
        all_errs.append(e)
        all_radii.append(rad)
    errs = np.concatenate(all_errs)
    radii = np.concatenate(all_radii)
    radial_bins = []
    r_max = float(radii.max())
    edges = np.linspace(0, r_max, 7)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (radii >= lo) & (radii < hi)
        if m.any():
            radial_bins.append({
                "r_lo": float(lo), "r_hi": float(hi),
                "n": int(m.sum()), "mean_err": float(errs[m].mean()),
                "median_err": float(np.median(errs[m])),
            })

    result = {
        "method": method_name,
        "params": method_params or {},
        "status": "ok",
        "n_views": len(obj_views),
        "n_corners": int(errs.size),
        "image_size": list(image_size),
        "rms": float(rms),
        "K": K.tolist(),
        "D": D.ravel().tolist(),
        "per_view_rms_mean": float(np.mean(per_view_rms)),
        "per_view_rms_median": float(np.median(per_view_rms)),
        "per_view_rms_p95": float(np.percentile(per_view_rms, 95)),
        "radial_bins": radial_bins,
        "t_detect_s": round(t_detect, 1),
        "t_calibrate_s": round(t_calibrate, 1),
        "tag_corner_shift": TAG_CORNER_SHIFT,
        "fix_aspect_ratio": fix_aspect_ratio,
        "rational_model": rational_model,
        "pinhole_size": list(pinhole_size),
        "fov_deg": fov_deg,
        "outlier_filter_pct": outlier_filter_pct,
        "dropped_views": dropped_views,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _append_result(result)
    return result


def _append_result(result: dict) -> None:
    RESULTS_LOG.touch(exist_ok=True)
    with RESULTS_LOG.open("a") as f:
        f.write(json.dumps(result) + "\n")
    print(f"[{result['method']}] {result.get('status')}  RMS={result.get('rms', '-')}")


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------


METHODS: dict[str, DetectFn] = {
    "baseline_aruco_subpix": m_baseline_aruco_subpix,
    "aruco_apriltag_refine": m_aruco_apriltag_refine,
    "pupil_apriltags_default": m_pupil_apriltags_default,
    "aruco_norefine": m_aruco_norefine,
    "aruco_contour": m_aruco_contour,
    "aruco_subpix_w3": m_aruco_subpix_w3,
    "aruco_subpix_w7": m_aruco_subpix_w7,
    "aruco_subpix_w11": m_aruco_subpix_w11,
    "aruco_subpix_w7_tight": m_aruco_subpix_w7_tight,
    "clahe_then_subpix": m_clahe_then_subpix,
    "upscale2x_subpix": m_upscale2x_subpix,
    "upscale2x_norefine": m_upscale2x_norefine,
    "upscale4x_norefine": m_upscale4x_norefine,
    "upscale8x_norefine": m_upscale8x_norefine,
    "upscale16x_norefine": m_upscale16x_norefine,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", action="append", default=[],
                    help="name of a method (repeatable; will run all in order)")
    ap.add_argument("--methods", help="comma-separated list, equivalent to repeated --method")
    ap.add_argument("--list", action="store_true", help="list registered methods and exit")
    ap.add_argument("--frame-stride", type=int, default=20,
                    help="sample every N-th frame (default 20 = ~1100 frames "
                    "of the 22k-frame video, ~5 min per iteration)")
    ap.add_argument("--rational-model", action="store_true")
    ap.add_argument("--pinhole-size", type=int, default=PINHOLE_SIZE[0],
                    help="square unwrap resolution (default 1280)")
    ap.add_argument("--build-cache", action="store_true",
                    help="prepass: decode + unwrap every Nth video frame to "
                    "an .npy cache on disk; subsequent --method runs read "
                    "from cache instead of re-decoding the H.265 source. "
                    "~10x speedup per iteration.")
    ap.add_argument("--rebuild-cache", action="store_true",
                    help="rebuild even if a cache file already exists")
    ap.add_argument("--no-cache", action="store_true",
                    help="ignore any cache; always read from video")
    ap.add_argument("--outlier-filter-pct", type=float, default=0.0,
                    help="after first calibrate, drop top X%% of views by "
                    "per-view RMS and recalibrate")
    args = ap.parse_args()

    if args.list:
        for name in METHODS:
            print(name)
        return

    pinhole_size = (args.pinhole_size, args.pinhole_size)

    if args.build_cache or args.rebuild_cache:
        build_unwrap_cache(
            pinhole_size=pinhole_size,
            fov_deg=FOV_DEG,
            frame_stride=args.frame_stride,
            overwrite=args.rebuild_cache,
        )
        if not (args.method or args.methods):
            return  # --build-cache alone is fine

    methods = list(args.method)
    if args.methods:
        methods += args.methods.split(",")
    if not methods:
        ap.error("--method, --methods, --list, or --build-cache required")
    for m in methods:
        if m not in METHODS:
            ap.error(f"unknown method '{m}'; --list shows registered ones")

    method_params = {"pinhole_size": list(pinhole_size)}
    for m in methods:
        print(f"\n========== {m}  (pinhole={pinhole_size}) ==========")
        run_benchmark(
            m, METHODS[m],
            method_params=method_params,
            frame_stride=args.frame_stride,
            rational_model=args.rational_model,
            pinhole_size=pinhole_size,
            use_cache=not args.no_cache,
            outlier_filter_pct=args.outlier_filter_pct,
        )


if __name__ == "__main__":
    main()
