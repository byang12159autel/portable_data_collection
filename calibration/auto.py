#!/usr/bin/env python3
"""Auto-sweep fisheye calibration: try progressively looser params, pick best.

Demuxes a ``.insv`` (or accepts pre-demuxed per-lens mp4s), collects every
plausible view once with permissive thresholds, then sweeps a grid of
``(frame_stride, min_corners_per_view, min_spread_frac)`` filters,
calibrates each lens with each combo, and saves the first result that
meets quality criteria — or the best by RMS if none does.

Quality criteria (configurable via ``--target-rms`` / ``--target-focal-range``):
    - RMS                          < 2.0 px
    - fx, fy                       within [400, 800] for a 1920x1920 lens
    - |fy/fx - 1|                  < 0.15
    - |D[0]| (k1)                  < 1.0

Usage::

    pixi run python -m calibration.auto \\
        --insv data/insta360_calibration/VID_20260515_142743_00_002.insv \\
        --marker-config config/apriltag_board.yaml \\
        --output data/insta360_intrinsics.npz

Or if you've already demuxed::

    pixi run python -m calibration.auto \\
        --front-video data/..._lens0.mp4 \\
        --back-video  data/..._lens1.mp4 \\
        --marker-config config/apriltag_board.yaml \\
        --output data/insta360_intrinsics.npz
"""

from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path  # noqa: TC003 — tyro needs Path at runtime
from typing import Literal

import cv2
import numpy as np
import tyro

from calibration.fisheye import (
    LensCalibration,
    _calibrate_one_lens,
    _can_calibrate_single_view,
    _well_spread,
    build_extractor,
    save_intrinsics,
)
from core.camera.convert import convert as convert_insv

TargetType = Literal["charuco", "apriltag_grid"]


@dataclasses.dataclass
class Args:
    """Auto-sweep fisheye calibration."""

    marker_config: Path
    """YAML with a ``charuco:`` or ``apriltag_grid:`` section."""

    output: Path
    """Destination .npz."""

    insv: Path | None = None
    """Source .insv file. Will be demuxed into <stem>_lens0/1.mp4 next to it
    unless --front-video / --back-video are also given."""

    front_video: Path | None = None
    """Per-lens recording of the front lens (e.g. lens0.mp4). Overrides --insv."""

    back_video: Path | None = None
    """Per-lens recording of the back lens (e.g. lens1.mp4). Overrides --insv."""

    target_type: TargetType = "apriltag_grid"
    """charuco | apriltag_grid."""

    target_rms: float = 2.0
    """Stop the sweep as soon as a result has RMS below this (px)."""

    focal_min: float = 400.0
    """Sanity check: fx and fy must exceed this for "acceptable"."""

    focal_max: float = 800.0
    """Sanity check: fx and fy must be below this for "acceptable"."""

    fy_fx_tolerance: float = 0.15
    """Sanity check: |fy/fx - 1| must be below this."""

    k1_max: float = 1.0
    """Sanity check: |D[0]| (the dominant fisheye distortion coeff) must be below this."""

    permissive_min_corners: int = 4
    """Initial-pass corner threshold (intentionally loose; sweep tightens it)."""


# Sweep grid: each tuple is (frame_stride, min_corners_per_view, min_spread_frac).
# Ordered from tightest -> loosest. We stop on the first "acceptable" result.
SWEEP_GRID: list[tuple[int, int, float]] = [
    (5, 24, 0.20),
    (5, 16, 0.15),
    (2, 16, 0.15),
    (2, 12, 0.10),
    (1, 12, 0.10),
    (1, 8, 0.05),
    (1, 6, 0.03),
]


# ---------------------------------------------------------------------------
# Permissive collection (run once per lens)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _CollectedViews:
    obj_views: list[np.ndarray]
    img_views: list[np.ndarray]
    image_size: tuple[int, int]


def collect_candidate_views(
    video: Path, extract, permissive_min_corners: int, label: str,
) -> _CollectedViews:
    """Walk the entire video once with very loose filtering.

    Drops only:
      - frames where the detector returned None
      - frames that fail the single-view fisheye-calibrate trial (truly degenerate)

    Result is the maximal candidate pool; downstream sweep applies the
    expensive thresholds (min_corners, min_spread) in memory.
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video}")
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n[{label}] {video}  ({n_total} frames) — permissive collection pass ...")

    obj_views: list[np.ndarray] = []
    img_views: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    K_init: np.ndarray | None = None
    n_dropped_trial = 0
    idx = -1

    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        idx += 1
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if image_size is None:
            h, w = rgb.shape[:2]
            image_size = (w, h)
            f_init = w / math.pi
            K_init = np.array(
                [[f_init, 0.0, w / 2.0],
                 [0.0, f_init, h / 2.0],
                 [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            print(f"[{label}] image size: {w}x{h}")

        res = extract(rgb)
        if res is None:
            continue
        assert K_init is not None
        # Single-view trial is the authoritative degeneracy filter; tight-spread
        # and few-corner views may still be useful for some sweep combos.
        if not _can_calibrate_single_view(res[0], res[1], image_size, K_init):
            n_dropped_trial += 1
            continue
        obj_views.append(res[0])
        img_views.append(res[1])

        if idx % 200 == 0:
            print(f"[{label}]  frame {idx}/{n_total}  candidates={len(obj_views)}")

    cap.release()
    if image_size is None:
        raise RuntimeError(f"{video} had no readable frames")

    print(
        f"[{label}] collected {len(obj_views)} candidate views "
        f"(dropped {n_dropped_trial} for failing single-view trial)"
    )
    return _CollectedViews(obj_views, img_views, image_size)


# ---------------------------------------------------------------------------
# Acceptance check
# ---------------------------------------------------------------------------


def _is_acceptable(
    cal: LensCalibration,
    target_rms: float,
    focal_min: float,
    focal_max: float,
    fy_fx_tol: float,
    k1_max: float,
) -> tuple[bool, str]:
    fx, fy = float(cal.K[0, 0]), float(cal.K[1, 1])
    k1 = float(cal.D.ravel()[0])
    if not math.isfinite(cal.rms) or cal.rms > target_rms:
        return False, f"RMS {cal.rms:.2f} > {target_rms}"
    if not (focal_min <= fx <= focal_max and focal_min <= fy <= focal_max):
        return False, f"focal out of [{focal_min}, {focal_max}]: fx={fx:.1f} fy={fy:.1f}"
    if abs(fy / fx - 1.0) > fy_fx_tol:
        return False, f"fy/fx={fy / fx:.3f} too far from 1.0"
    if abs(k1) > k1_max:
        return False, f"|k1|={abs(k1):.3f} > {k1_max}"
    return True, "ok"


# ---------------------------------------------------------------------------
# Sweep over filter combos
# ---------------------------------------------------------------------------


def _sweep_one_lens(
    pool: _CollectedViews,
    label: str,
    args: Args,
) -> LensCalibration | None:
    """Try each sweep combo on the candidate pool. Return the first acceptable
    result, or the best-by-RMS if none is acceptable, or None if all failed."""
    best: LensCalibration | None = None
    best_combo: tuple | None = None
    best_status = ""

    print(f"\n[{label}] sweeping {len(SWEEP_GRID)} parameter combinations ...")
    for stride, min_corners, min_spread in SWEEP_GRID:
        # Apply this combo's filters to the candidate pool.
        obj_filtered: list[np.ndarray] = []
        img_filtered: list[np.ndarray] = []
        for i, (obj, img) in enumerate(zip(pool.obj_views, pool.img_views)):
            if i % stride != 0:
                continue
            if img.shape[0] < min_corners:
                continue
            if not _well_spread(img, pool.image_size, min_spread):
                continue
            obj_filtered.append(obj)
            img_filtered.append(img)

        combo_label = f"stride={stride} min_corners={min_corners} min_spread={min_spread:.2f}"
        if len(obj_filtered) < 5:
            print(f"[{label}] {combo_label}: only {len(obj_filtered)} views, skipping")
            continue

        print(f"\n[{label}] {combo_label}: {len(obj_filtered)} views")
        try:
            K, D, rms, n_used = _calibrate_one_lens(
                obj_filtered, img_filtered, pool.image_size, label,
            )
        except cv2.error as e:
            print(f"[{label}] {combo_label}: calibrate failed ({e})")
            continue

        cal = LensCalibration(K=K, D=D, rms=rms, n_views=n_used, image_size=pool.image_size)
        ok, status = _is_acceptable(
            cal,
            target_rms=args.target_rms,
            focal_min=args.focal_min,
            focal_max=args.focal_max,
            fy_fx_tol=args.fy_fx_tolerance,
            k1_max=args.k1_max,
        )
        print(f"[{label}] {combo_label}: RMS={rms:.3f} px, status={status}")

        if ok:
            print(f"[{label}] ACCEPTED at {combo_label}")
            return cal

        if best is None or rms < best.rms:
            best = cal
            best_combo = (stride, min_corners, min_spread)
            best_status = status

    if best is not None:
        s, c, p = best_combo  # type: ignore[misc]
        print(
            f"\n[{label}] no combo met acceptance; best was "
            f"stride={s} min_corners={c} min_spread={p:.2f}  "
            f"RMS={best.rms:.3f} px  ({best_status})"
        )
    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: Args) -> None:
    # Resolve per-lens video paths.
    front = args.front_video
    back = args.back_video
    if args.insv is not None and (front is None or back is None):
        if not args.insv.is_file():
            print(f"Error: {args.insv} not found", file=sys.stderr)
            sys.exit(1)
        out0 = args.insv.with_name(args.insv.stem + "_lens0.mp4")
        out1 = args.insv.with_name(args.insv.stem + "_lens1.mp4")
        if not (out0.exists() and out1.exists()):
            print(f"Demuxing {args.insv} ...")
            convert_insv(args.insv, out0, out1, force=False)
        front = front or out0
        back = back or out1

    if front is None and back is None:
        print("Error: provide --insv or --front-video / --back-video", file=sys.stderr)
        sys.exit(1)

    # Build the per-frame extractor with the most permissive corner threshold;
    # the sweep applies tighter thresholds post-hoc.
    try:
        extract = build_extractor(args.target_type, args.marker_config, args.permissive_min_corners)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    results: dict[str, LensCalibration] = {}
    for label, video in [("front", front), ("back", back)]:
        if video is None:
            continue
        if not video.is_file():
            print(f"[{label}] {video} not found, skipping")
            continue
        pool = collect_candidate_views(video, extract, args.permissive_min_corners, label)
        if not pool.obj_views:
            print(f"[{label}] no candidate views; skipping")
            continue
        cal = _sweep_one_lens(pool, label, args)
        if cal is not None:
            results[label] = cal

    if not results:
        print("\nError: no lens produced any calibration result", file=sys.stderr)
        sys.exit(1)

    save_intrinsics(args.output, results)
    print(f"\nWrote {args.output}")
    print("\nFinal:")
    for label, cal in results.items():
        fx, fy = float(cal.K[0, 0]), float(cal.K[1, 1])
        print(
            f"  [{label}]  RMS={cal.rms:.3f} px  fx={fx:.1f} fy={fy:.1f}  "
            f"D={cal.D.ravel().tolist()}  ({cal.n_views} views)"
        )


if __name__ == "__main__":
    main(tyro.cli(Args))
