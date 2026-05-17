#!/usr/bin/env python3
"""Two-stage calibration: equidistant unwrap + pinhole-model refine.

When ``cv2.fisheye.calibrate`` fails (typically because the calibration
recording doesn't get corners to the fisheye periphery), this is the
fallback. We sidestep fisheye-model fitting entirely:

1. **Stage 1 — equidistant unwrap.** Each frame is rectified to a
   virtual pinhole using the equidistant fisheye assumption
   (``f = W/pi``, principal point at image centre, zero distortion).
   The result is *approximately* a pinhole image; deviations from
   equidistant show up as residual radial distortion.
2. **Stage 2 — pinhole refine.** ``cv2.calibrateCamera`` (the standard
   pinhole model, not the fisheye one) is run on the board corners
   detected in the rough-pinhole frames. The 5-coefficient pinhole
   distortion model captures the residual equidistant-vs-actual
   mismatch.

The combined model (equidistant unwrap + pinhole undistort) is what
the downstream rectifier composes into a single LUT per lens.

Why this works when ``cv2.fisheye.calibrate`` doesn't:

- The fisheye unwrap is fixed and parameter-free (no optimization on
  the slippery k1..k4 fisheye coefficients).
- The pinhole calibration solves a 9-parameter problem (fx, fy, cx,
  cy, k1, k2, p1, p2, k3) with the much better-conditioned standard
  ``cv2.calibrateCamera`` solver. It tolerates limited coverage far
  better than the fisheye optimizer.
- After unwrap, the board's image-space spread is no worse than in the
  original fisheye, but the model the data has to constrain is simpler.

Trade-off: the chosen FOV and output size are *baked in* at calibration
time — running rectify with a different FOV requires re-running this
script. Default is 110° at 1280×1280, matching ``rectify.py``.

Usage::

    pixi run python -m pose_calibration.calibration.two_stage \\
        --front-video data/<recording>_lens0.mp4 \\
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

from pose_calibration.calibration.rectify import Rectifier, _pinhole_K
from pose_calibration.markers.detect import (
    apriltag_grid_object_points,
    charuco_object_points,
    create_charuco_board,
    detect_aruco_markers,
    detect_charuco_corners,
    load_apriltag_grid_configs,
    load_charuco_board_configs,
)

TargetType = Literal["charuco", "apriltag_grid"]


@dataclasses.dataclass
class Args:
    """Two-stage fisheye-then-pinhole calibration."""

    marker_config: Path
    """YAML with a ``charuco:`` or ``apriltag_grid:`` section."""

    output: Path
    """Destination .npz. Existing per-lens entries are preserved if not re-run."""

    front_video: Path | None = None
    """Per-lens recording of the front lens."""

    back_video: Path | None = None
    """Per-lens recording of the back lens."""

    target_type: TargetType = "apriltag_grid"
    """charuco | apriltag_grid."""

    fov_deg: float = 110.0
    """Virtual-pinhole FOV used for stage 1 unwrap (and baked into the result)."""

    pinhole_width: int = 1280
    """Stage 1 output width."""

    pinhole_height: int = 1280
    """Stage 1 output height."""

    frame_stride: int = 2
    """Use every Nth frame for calibration."""

    min_corners_per_view: int = 8
    """Drop a view with fewer than this many detected corners (post-unwrap)."""

    min_views: int = 10
    """Minimum frames with detections to attempt pinhole calibration."""

    free_K: bool = False
    """Allow cv2.calibrateCamera to refine fx, fy, cx, cy. Off by default —
    when the recording is sparse, freeing K causes the optimizer to collapse
    focal length (same failure mode as fisheye calibration). Keeping K fixed
    to the equidistant-derived value and letting only the radial distortion
    coefficients refine is much more robust with limited data."""

    fix_k3: bool = True
    """Hold k3 at zero. High-order radial distortion needs corners at large
    image radii to be identifiable; with marginal coverage, freeing k3
    overfits."""


# ---------------------------------------------------------------------------
# Per-frame board extraction in rough-pinhole coordinates
# ---------------------------------------------------------------------------


def _build_pinhole_extractor(args: Args):
    """Return an ``extract(rgb_pinhole) -> (obj_pts, img_pts) | None`` closure."""
    if args.target_type == "charuco":
        boards = load_charuco_board_configs(args.marker_config)
        if not boards:
            raise RuntimeError(f"no charuco board in {args.marker_config}")
        name, cfg = next(iter(boards.items()))
        board = create_charuco_board(cfg)
        print(f"Target: ChArUco '{name}' ({cfg.squares_x}x{cfg.squares_y})")

        def extract(rgb: np.ndarray):
            ch_corners, ch_ids, _, _ = detect_charuco_corners(
                rgb, board, min_corners=args.min_corners_per_view
            )
            if ch_corners is None or ch_ids is None:
                return None
            obj = charuco_object_points(board, ch_ids).astype(np.float32).reshape(-1, 1, 3)
            img = ch_corners.reshape(-1, 1, 2).astype(np.float32)
            return obj, img

        return extract

    if args.target_type == "apriltag_grid":
        grids = load_apriltag_grid_configs(args.marker_config)
        if not grids:
            raise RuntimeError(f"no apriltag_grid in {args.marker_config}")
        name, grid_cfg = next(iter(grids.items()))
        dict_id = grid_cfg.cv2_dictionary
        allowed_ids = set(grid_cfg.tag_ids)
        grid_obj_pts = apriltag_grid_object_points(grid_cfg)
        print(
            f"Target: AprilGrid '{name}' ({grid_cfg.tag_cols}x{grid_cfg.tag_rows}, "
            f"{grid_cfg.dictionary})"
        )

        def extract(rgb: np.ndarray):
            corners, ids = detect_aruco_markers(
                rgb, marker_dict=dict_id, allowed_ids=allowed_ids
            )
            if ids is None:
                return None
            obj_list: list[np.ndarray] = []
            img_list: list[np.ndarray] = []
            for c, tid in zip(corners, ids.flatten()):
                tid = int(tid)
                if tid not in grid_obj_pts:
                    continue
                obj_list.append(grid_obj_pts[tid])
                img_list.append(c.reshape(4, 2))
            if not obj_list or sum(len(p) for p in obj_list) < args.min_corners_per_view:
                return None
            obj = np.vstack(obj_list).astype(np.float32).reshape(-1, 1, 3)
            img = np.vstack(img_list).astype(np.float32).reshape(-1, 1, 2)
            return obj, img

        return extract

    raise ValueError(args.target_type)


# ---------------------------------------------------------------------------
# Per-lens two-stage pass
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TwoStageResult:
    """Outputs for one lens."""

    K_fisheye: np.ndarray   # equidistant K used for stage 1 (3, 3)
    D_fisheye: np.ndarray   # zeros (4,) — equidistant has no fisheye D
    image_size: tuple[int, int]  # (W, H) of the raw fisheye

    K_pinhole_rough: np.ndarray   # K of the stage 1 output (3, 3)
    K_pinhole_refined: np.ndarray  # K returned by cv2.calibrateCamera (3, 3)
    D_pinhole_refined: np.ndarray  # standard pinhole distortion (5,)
    pinhole_size: tuple[int, int]  # (W, H) of stage 1 output
    rms: float
    n_views: int


def _calibrate_one_lens(
    video: Path,
    extract,
    fov_deg: float,
    pinhole_size: tuple[int, int],
    frame_stride: int,
    min_views: int,
    label: str,
    free_K: bool,
    fix_k3: bool,
) -> TwoStageResult | None:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video}")
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n[{label}] {video}  ({n_total} frames)")

    # Stage 1 rectifier built lazily once we see the first frame.
    rectifier: Rectifier | None = None
    fisheye_size: tuple[int, int] | None = None
    K_fisheye: np.ndarray | None = None

    obj_views: list[np.ndarray] = []
    img_views: list[np.ndarray] = []
    idx = -1
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        idx += 1
        if idx % frame_stride != 0:
            continue

        if rectifier is None:
            h, w = bgr.shape[:2]
            fisheye_size = (w, h)
            f_eq = w / math.pi
            K_fisheye = np.array(
                [[f_eq, 0.0, w / 2.0],
                 [0.0, f_eq, h / 2.0],
                 [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            rectifier = Rectifier.build(
                K_fisheye, np.zeros(4, dtype=np.float64), pinhole_size, fov_deg,
            )
            print(
                f"[{label}] fisheye {w}x{h} -> rough pinhole "
                f"{pinhole_size[0]}x{pinhole_size[1]} @ {fov_deg:.1f}deg"
            )

        rough_bgr = rectifier.apply(bgr)
        rough_rgb = cv2.cvtColor(rough_bgr, cv2.COLOR_BGR2RGB)
        res = extract(rough_rgb)
        if res is None:
            continue
        obj_views.append(res[0])
        img_views.append(res[1])

        if idx % 200 == 0:
            print(f"[{label}]  frame {idx}/{n_total}  views={len(obj_views)}")

    cap.release()

    if rectifier is None or fisheye_size is None or K_fisheye is None:
        print(f"[{label}] no frames decoded; skipping")
        return None

    print(f"[{label}] collected {len(obj_views)} pinhole views")
    if len(obj_views) < min_views:
        print(
            f"[{label}] only {len(obj_views)} views (need >= {min_views}); skipping this lens",
            file=sys.stderr,
        )
        return None

    # K is fixed to the equidistant-derived rough-pinhole K unless --free-K.
    # This matches the GS-community convention: trust the unwrap's K, model
    # the lens's deviation from equidistant as radial distortion.
    flags = cv2.CALIB_USE_INTRINSIC_GUESS
    if not free_K:
        flags |= cv2.CALIB_FIX_FOCAL_LENGTH | cv2.CALIB_FIX_PRINCIPAL_POINT
    flags |= cv2.CALIB_ZERO_TANGENT_DIST  # p1=p2=0; fisheye unwrap has no tangential component
    if fix_k3:
        flags |= cv2.CALIB_FIX_K3

    free_params = []
    if free_K:
        free_params.append("K")
    free_params.extend(["k1", "k2"])
    if not fix_k3:
        free_params.append("k3")
    print(
        f"[{label}] running cv2.calibrateCamera (standard pinhole, "
        f"refining: {', '.join(free_params)}) ..."
    )

    rms, K_refined, D_refined, _, _ = cv2.calibrateCamera(
        obj_views,
        img_views,
        pinhole_size,
        rectifier.K_pinhole.copy(),
        np.zeros(5, dtype=np.float64),
        flags=flags,
    )
    rms = float(rms)
    print(f"[{label}] RMS = {rms:.4f} px")
    print(
        f"[{label}] K_refined: fx={K_refined[0, 0]:.1f} fy={K_refined[1, 1]:.1f} "
        f"cx={K_refined[0, 2]:.1f} cy={K_refined[1, 2]:.1f}"
    )
    print(f"[{label}] D_refined: {D_refined.ravel().tolist()}")

    return TwoStageResult(
        K_fisheye=K_fisheye,
        D_fisheye=np.zeros(4, dtype=np.float64),
        image_size=fisheye_size,
        K_pinhole_rough=rectifier.K_pinhole.astype(np.float64),
        K_pinhole_refined=K_refined.astype(np.float64),
        D_pinhole_refined=D_refined.ravel().astype(np.float64),
        pinhole_size=pinhole_size,
        rms=rms,
        n_views=len(obj_views),
    )


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def _save(output: Path, results: dict[str, TwoStageResult]) -> None:
    existing: dict[str, np.ndarray] = {}
    if output.exists():
        existing = dict(np.load(str(output)))

    out: dict[str, np.ndarray] = dict(existing)
    image_size_seen: tuple[int, int] | None = None
    for label, r in results.items():
        out[f"K_{label}"] = r.K_fisheye
        out[f"D_{label}"] = r.D_fisheye
        out[f"rms_{label}"] = np.array(r.rms)
        out[f"n_{label}"] = np.array(r.n_views)

        out[f"K_{label}_pinhole_rough"] = r.K_pinhole_rough
        out[f"K_{label}_pinhole_refined"] = r.K_pinhole_refined
        out[f"D_{label}_pinhole_refined"] = r.D_pinhole_refined
        out[f"pinhole_size_{label}"] = np.array(r.pinhole_size, dtype=np.int32)
        image_size_seen = r.image_size

    if image_size_seen is None and "image_size" in existing:
        image_size_seen = tuple(int(v) for v in existing["image_size"])  # type: ignore[assignment]
    if image_size_seen is not None:
        out["image_size"] = np.array(image_size_seen, dtype=np.int32)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(output), **out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: Args) -> None:
    if args.front_video is None and args.back_video is None:
        print("Error: at least one of --front-video / --back-video is required", file=sys.stderr)
        sys.exit(1)

    extract = _build_pinhole_extractor(args)
    pinhole_size = (args.pinhole_width, args.pinhole_height)

    results: dict[str, TwoStageResult] = {}
    for label, video in [("front", args.front_video), ("back", args.back_video)]:
        if video is None:
            continue
        if not video.is_file():
            print(f"[{label}] {video} not found, skipping")
            continue
        r = _calibrate_one_lens(
            video, extract, args.fov_deg, pinhole_size,
            args.frame_stride, args.min_views, label,
            free_K=args.free_K, fix_k3=args.fix_k3,
        )
        if r is not None:
            results[label] = r

    if not results:
        print("\nError: no lens produced a calibration", file=sys.stderr)
        sys.exit(1)

    _save(args.output, results)
    print(f"\nWrote {args.output}")
    for label, r in results.items():
        fx, fy = float(r.K_pinhole_refined[0, 0]), float(r.K_pinhole_refined[1, 1])
        print(
            f"  [{label}] two-stage: RMS={r.rms:.3f} px  "
            f"fx={fx:.1f} fy={fy:.1f}  D={r.D_pinhole_refined.tolist()}  "
            f"({r.n_views} views)"
        )


if __name__ == "__main__":
    main(tyro.cli(Args))
