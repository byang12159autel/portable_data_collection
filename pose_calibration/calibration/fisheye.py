#!/usr/bin/env python3
"""Per-lens fisheye calibration from a board recording.

Each lens is its own video file (use ``insta360.convert`` first to
demux a ``.insv`` into ``_lens0.mp4`` and ``_lens1.mp4``). Pass either
or both — each lens is calibrated independently. If ``--output`` exists,
its existing entries for the other lens are preserved, so you can
calibrate one lens at a time across separate recordings.

Usage::

    # Calibrate both lenses from two per-lens recordings
    pixi run python -m pose_calibration.calibration.fisheye \\
        --front-video data/calib_lens0.mp4 \\
        --back-video data/calib_lens1.mp4 \\
        --marker-config config/apriltag_board.yaml \\
        --output data/insta360_intrinsics.npz

    # Calibrate just the front lens (re-run later for back)
    pixi run python -m pose_calibration.calibration.fisheye \\
        --front-video data/front_calib.mp4 \\
        --marker-config config/apriltag_board.yaml \\
        --output data/insta360_intrinsics.npz

Output ``.npz`` carries::

    K_front, D_front, K_back, D_back   # cv2.fisheye model (D is (4,))
    image_size                          # (W, H) of each lens crop
    rms_front, rms_back                 # calibration RMS reprojection error (px)
    n_front, n_back                     # number of frames used per lens
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
    """Per-lens fisheye calibration from a board recording."""

    marker_config: Path
    """YAML with a ``charuco:`` or ``apriltag_grid:`` section."""

    output: Path
    """Destination .npz. Existing entries for lenses not (re)calibrated this run are preserved."""

    front_video: Path | None = None
    """Per-lens recording of the front lens, e.g. lens0.mp4. Optional."""

    back_video: Path | None = None
    """Per-lens recording of the back lens, e.g. lens1.mp4. Optional."""

    target_type: TargetType = "apriltag_grid"
    """charuco | apriltag_grid."""

    frame_stride: int = 5
    """Use every Nth frame. Lower = more views, slower."""

    min_views: int = 10
    """Minimum frames with detections required to attempt calibration."""

    min_corners_per_view: int = 16
    """Drop a view if fewer than this many corners are detected. 16 = 4 AprilGrid tags."""

    min_spread_frac: float = 0.15
    """Drop a view if detected corners span < this fraction of the image in either
    axis. Co-linear / tightly-clustered detections trip cv2.fisheye.calibrate's
    extrinsic init solver, so we filter them out up front."""

    equidistant_only: bool = False
    """Skip OpenCV fisheye.calibrate; just write equidistant-model defaults
    (f = W/pi, principal point at image centre, D = 0). Use this to verify
    downstream rectify / detection plumbing before doing a real calibration.
    Rectified output and any downstream PnP poses will be approximate."""

    fix_K: bool = False
    """Lock fx, fy to W/pi (equidistant) and cx, cy to the image centre;
    only fit the 4 distortion coefficients (k1..k4). Use when the
    recording lacks distance variation — the optimizer otherwise
    collapses fx/fy. K is approximate but D captures the actual lens
    deviation; good enough for marker-pose work."""


# ---------------------------------------------------------------------------
# Per-frame corner extraction
# ---------------------------------------------------------------------------


def _extract_charuco(
    img_rgb: np.ndarray,
    board: cv2.aruco.CharucoBoard,
    min_corners: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    ch_corners, ch_ids, _, _ = detect_charuco_corners(img_rgb, board, min_corners=min_corners)
    if ch_corners is None or ch_ids is None or len(ch_ids) < min_corners:
        return None
    obj = charuco_object_points(board, ch_ids).astype(np.float64).reshape(-1, 1, 3)
    img = ch_corners.reshape(-1, 1, 2).astype(np.float64)
    return obj, img


def _extract_apriltag_grid(
    img_rgb: np.ndarray,
    dict_id: int,
    allowed_ids: set[int],
    grid_obj_pts: dict[int, np.ndarray],
    min_corners: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    corners, ids = detect_aruco_markers(
        img_rgb, marker_dict=dict_id, allowed_ids=allowed_ids
    )
    if ids is None:
        return None
    obj_list: list[np.ndarray] = []
    img_list: list[np.ndarray] = []
    for c, tid in zip(corners, ids.flatten()):
        tid = int(tid)
        if tid not in grid_obj_pts:
            continue
        obj_list.append(grid_obj_pts[tid])  # (4, 3) float32
        img_list.append(c.reshape(4, 2))     # (4, 2) float32
    if not obj_list or sum(len(p) for p in obj_list) < min_corners:
        return None
    obj = np.vstack(obj_list).astype(np.float64).reshape(-1, 1, 3)
    img = np.vstack(img_list).astype(np.float64).reshape(-1, 1, 2)
    return obj, img


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def _calibrate_one_lens(
    obj_pts: list[np.ndarray],
    img_pts: list[np.ndarray],
    image_size: tuple[int, int],
    lens_label: str,
    fix_K: bool = False,
    outlier_iters: int = 5,
    outlier_ratio: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Run cv2.fisheye.calibrate with crash recovery + per-view outlier rejection.

    Two failure modes are handled:
      - **Per-view crashes** (InitExtrinsics asserting on a view whose
        homography becomes degenerate at the current K). Linear-scan to
        find an offender, drop it, retry.
      - **Outlier views** that don't crash but contribute disproportionate
        residual (typically tag-rotation misdetection or planar-pose flip
        ambiguity on small detections). After each successful group
        calibration, compute each view's per-view RMS reprojection and
        drop any view whose RMS exceeds ``outlier_ratio * median``.
        Recalibrate. Iterate up to ``outlier_iters`` times or until no
        outliers remain.

    Returns (K, D, rms_px, n_views_used).
    """
    if not obj_pts:
        raise RuntimeError(f"{lens_label}: no views to calibrate")

    # Seed K from the equidistant fisheye model: r_image = f * theta with
    # theta_max = pi/2 mapping to (W/2), so f = W/pi.
    w, h = image_size
    f_init = w / math.pi
    K_init = np.array(
        [[f_init, 0.0, w / 2.0],
         [0.0, f_init, h / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    # CALIB_RECOMPUTE_EXTRINSIC is deliberately omitted: it re-runs the
    # homography-based InitExtrinsics on every LM iteration with the
    # current (drifting) K, D. A view that's well-conditioned at K_init
    # can become degenerate at a different K and crash. Without this flag
    # InitExtrinsics runs only once at start, and extrinsics refine via
    # gradient descent. The single-view pre-filter has already verified
    # all views survive InitExtrinsics at K_init.
    flags = (
        cv2.fisheye.CALIB_FIX_SKEW
        | cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
    )
    if fix_K:
        # Lock K to the equidistant seed; only the 4 distortion coeffs refine.
        # Use this when the recording lacks distance/depth variation so the
        # optimizer can't separate focal length from per-view translation.
        flags |= cv2.fisheye.CALIB_FIX_FOCAL_LENGTH | cv2.fisheye.CALIB_FIX_PRINCIPAL_POINT
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    obj = list(obj_pts)
    img = list(img_pts)

    def run_group(obj_l: list[np.ndarray], img_l: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, float, list, list]:
        """Run cv2.fisheye.calibrate with crash recovery. Returns K, D, rms, rvecs, tvecs."""
        cur_obj = list(obj_l)
        cur_img = list(img_l)
        dropped = 0
        while True:
            K = K_init.copy()
            D = np.zeros(4, dtype=np.float64)
            rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(len(cur_obj))]
            tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(len(cur_obj))]
            try:
                rms, _, _, _, _ = cv2.fisheye.calibrate(
                    cur_obj, cur_img, image_size, K, D, rvecs, tvecs, flags, criteria,
                )
                if dropped:
                    print(
                        f"[{lens_label}]   group calibrate succeeded after dropping "
                        f"{dropped} crash-inducing view(s)"
                    )
                return K, D, float(rms), rvecs, tvecs
            except cv2.error as e:
                if len(cur_obj) <= 1:
                    raise
                bad_idx = _find_bad_view(
                    cur_obj, cur_img, image_size, K_init, flags, criteria,
                )
                if bad_idx is None:
                    raise RuntimeError(
                        f"{lens_label}: group calibration failed but no single bad view "
                        f"identified (error: {e})"
                    ) from e
                cur_obj.pop(bad_idx)
                cur_img.pop(bad_idx)
                dropped += 1

    # Outer loop: iterative outlier rejection on per-view reprojection RMS.
    K, D, rms, rvecs, tvecs = run_group(obj, img)
    for outlier_iter in range(outlier_iters):
        per_view = []
        for o, p, rv, tv in zip(obj, img, rvecs, tvecs):
            projected, _ = cv2.fisheye.projectPoints(
                o.astype(np.float64).reshape(-1, 1, 3), rv, tv, K, D,
            )
            residuals = projected.reshape(-1, 2) - p.astype(np.float64).reshape(-1, 2)
            per_view.append(float(np.sqrt(np.mean(np.sum(residuals * residuals, axis=1)))))
        per_view_arr = np.array(per_view)
        median = float(np.median(per_view_arr))
        threshold = max(outlier_ratio * median, 3.0)  # absolute floor of 3 px
        keep_mask = per_view_arr <= threshold
        n_keep = int(keep_mask.sum())
        n_drop = len(per_view_arr) - n_keep

        if n_drop == 0:
            break
        if n_keep < 10:
            print(
                f"[{lens_label}]   outlier iter {outlier_iter + 1}: would drop "
                f"{n_drop} but {n_keep} views would remain; stopping"
            )
            break

        print(
            f"[{lens_label}]   outlier iter {outlier_iter + 1}: "
            f"per-view RMS median={median:.2f} px, threshold={threshold:.2f} px, "
            f"dropping {n_drop}/{len(per_view_arr)} views (current group RMS={rms:.2f})"
        )
        obj = [obj[i] for i in range(len(obj)) if keep_mask[i]]
        img = [img[i] for i in range(len(img)) if keep_mask[i]]
        K, D, rms, rvecs, tvecs = run_group(obj, img)

    return K, D, rms, len(obj)


def _find_bad_view(
    obj: list[np.ndarray],
    img: list[np.ndarray],
    image_size: tuple[int, int],
    K_init: np.ndarray,
    flags: int,
    criteria: tuple,
) -> int | None:
    """Linear scan: return the index of a view whose removal lets calibrate succeed."""
    for i in range(len(obj)):
        trial_obj = obj[:i] + obj[i + 1 :]
        trial_img = img[:i] + img[i + 1 :]
        K = K_init.copy()
        D = np.zeros(4, dtype=np.float64)
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(len(trial_obj))]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(len(trial_obj))]
        try:
            cv2.fisheye.calibrate(
                trial_obj, trial_img, image_size, K, D, rvecs, tvecs, flags, criteria,
            )
            return i
        except cv2.error:
            continue
    return None


# ---------------------------------------------------------------------------
# Single-video → views
# ---------------------------------------------------------------------------


def _well_spread(img_pts: np.ndarray, image_size: tuple[int, int], min_frac: float) -> bool:
    """Reject views whose corners cluster in a thin pixel-space region."""
    pts = img_pts.reshape(-1, 2)
    w, h = image_size
    extent_x = float(pts[:, 0].max() - pts[:, 0].min())
    extent_y = float(pts[:, 1].max() - pts[:, 1].min())
    return extent_x > min_frac * w and extent_y > min_frac * h


def _can_calibrate_single_view(
    obj_pts: np.ndarray,
    img_pts: np.ndarray,
    image_size: tuple[int, int],
    K_init: np.ndarray,
) -> bool:
    """Authoritative degeneracy test: try ``cv2.fisheye.calibrate`` on this view alone.

    Calls calibrate with ``CALIB_FIX_INTRINSIC`` so it only solves for the view's
    6-DoF extrinsics. If that call survives, the view's homography is
    well-conditioned enough to also survive in the group calibration. If it
    raises (typically ``InitExtrinsics`` asserting on a degenerate
    homography), the view is unusable.

    Cost: one ``fisheye.calibrate`` invocation per candidate view, ~10-30 ms
    each. Worth it because it perfectly matches what the group calibrate
    will reject.
    """
    K = K_init.astype(np.float64).copy()
    D = np.zeros(4, dtype=np.float64)
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64)]
    try:
        cv2.fisheye.calibrate(
            [obj_pts.astype(np.float64).reshape(-1, 1, 3)],
            [img_pts.astype(np.float64).reshape(-1, 1, 2)],
            image_size, K, D, rvecs, tvecs,
            cv2.fisheye.CALIB_FIX_INTRINSIC | cv2.fisheye.CALIB_FIX_SKEW,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 1e-3),
        )
        return True
    except cv2.error:
        return False


def _collect_views(
    video: Path,
    extract,
    frame_stride: int,
    min_spread_frac: float,
    label: str,
) -> tuple[list[np.ndarray], list[np.ndarray], tuple[int, int]]:
    """Walk one per-lens video and return (obj_pts, img_pts, image_size)."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video}")
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n[{label}] {video}  ({n_total} frames)")

    obj_views: list[np.ndarray] = []
    img_views: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    K_init: np.ndarray | None = None
    n_dropped_spread = 0
    n_dropped_cond = 0

    idx = -1
    sampled = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        idx += 1
        if idx % frame_stride != 0:
            continue
        sampled += 1
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
        if res is not None:
            assert K_init is not None
            if not _well_spread(res[1], image_size, min_spread_frac):
                n_dropped_spread += 1
            elif not _can_calibrate_single_view(res[0], res[1], image_size, K_init):
                n_dropped_cond += 1
            else:
                obj_views.append(res[0])
                img_views.append(res[1])

        if sampled % 50 == 0:
            print(f"[{label}]  frame {idx}/{n_total}  views={len(obj_views)}")

    cap.release()
    if image_size is None:
        raise RuntimeError(f"{video} had no readable frames")

    print(
        f"[{label}] sampled {sampled} frames (stride {frame_stride}) -> "
        f"{len(obj_views)} usable views "
        f"(dropped {n_dropped_spread} for tight spread, "
        f"{n_dropped_cond} for failing single-view calibrate)"
    )
    return obj_views, img_views, image_size


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _equidistant_intrinsics(image_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """K, D for the idealised equidistant fisheye model: r = (W/pi) * theta."""
    w, h = image_size
    f = w / math.pi
    K = np.array(
        [[f, 0.0, w / 2.0],
         [0.0, f, h / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    D = np.zeros(4, dtype=np.float64)
    return K, D


def _peek_image_size(video: Path) -> tuple[int, int]:
    """Read the first frame of ``video`` and return its (W, H)."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video}")
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"{video} had no readable frames")
    h, w = bgr.shape[:2]
    return (w, h)


@dataclasses.dataclass
class LensCalibration:
    """One-lens calibration result; populated fields are valid if rms is finite."""

    K: np.ndarray
    D: np.ndarray
    rms: float
    n_views: int
    image_size: tuple[int, int]


ExtractFn = "callable"


def build_extractor(
    target_type: TargetType,
    marker_config: Path,
    min_corners_per_view: int,
) -> ExtractFn:
    """Build the per-frame corner-extraction closure used by `calibrate_video`."""
    if target_type == "charuco":
        boards = load_charuco_board_configs(marker_config)
        if not boards:
            raise RuntimeError(f"no charuco board in {marker_config}")
        name, cfg = next(iter(boards.items()))
        board = create_charuco_board(cfg)
        print(f"Target: ChArUco '{name}' ({cfg.squares_x}x{cfg.squares_y})")

        def extract(rgb: np.ndarray):
            return _extract_charuco(rgb, board, min_corners_per_view)

        return extract

    if target_type == "apriltag_grid":
        grids = load_apriltag_grid_configs(marker_config)
        if not grids:
            raise RuntimeError(f"no apriltag_grid in {marker_config}")
        name, grid_cfg = next(iter(grids.items()))
        dict_id = grid_cfg.cv2_dictionary
        allowed_ids = set(grid_cfg.tag_ids)
        grid_obj_pts = apriltag_grid_object_points(grid_cfg)
        print(
            f"Target: AprilGrid '{name}' ({grid_cfg.tag_cols}x{grid_cfg.tag_rows}, "
            f"{grid_cfg.dictionary})"
        )

        def extract(rgb: np.ndarray):
            return _extract_apriltag_grid(
                rgb, dict_id, allowed_ids, grid_obj_pts, min_corners_per_view
            )

        return extract

    raise ValueError(target_type)


def calibrate_video(
    video: Path,
    label: str,
    extract: ExtractFn,
    frame_stride: int,
    min_spread_frac: float,
    min_views: int,
    fix_K: bool = False,
) -> LensCalibration | None:
    """Walk ``video``, collect usable views, run fisheye calibration on them.

    Returns ``None`` if we couldn't gather enough views; raises ``cv2.error``
    only on solver errors that survive the in-script recovery loop (rare,
    only if the data is fundamentally unfittable).
    """
    obj_views, img_views, image_size = _collect_views(
        video, extract, frame_stride, min_spread_frac, label
    )
    if len(obj_views) < min_views:
        print(
            f"[{label}] only {len(obj_views)} views (need >= {min_views}); skipping this lens",
            file=sys.stderr,
        )
        return None
    print(
        f"\n[{label}] calibrating with {len(obj_views)} views "
        f"{'(K fixed to equidistant)' if fix_K else '(K free)'} ..."
    )
    K, D, rms, n_used = _calibrate_one_lens(
        obj_views, img_views, image_size, label, fix_K=fix_K,
    )
    print(f"[{label}] RMS = {rms:.4f} px  ({n_used} views used)")
    print(f"[{label}] fx={K[0, 0]:.1f} fy={K[1, 1]:.1f} cx={K[0, 2]:.1f} cy={K[1, 2]:.1f}")
    print(f"[{label}] D = {D.ravel()}")
    return LensCalibration(K=K, D=D, rms=rms, n_views=n_used, image_size=image_size)


def save_intrinsics(
    output: Path,
    results: dict[str, LensCalibration],
) -> None:
    """Merge ``results`` into ``output`` .npz, preserving entries for lenses we didn't touch."""
    existing: dict[str, np.ndarray] = {}
    if output.exists():
        existing = dict(np.load(str(output)))

    out: dict[str, np.ndarray] = dict(existing)
    image_sizes_seen: set[tuple[int, int]] = set()
    for label, cal in results.items():
        out[f"K_{label}"] = cal.K
        out[f"D_{label}"] = cal.D
        out[f"rms_{label}"] = np.array(cal.rms)
        out[f"n_{label}"] = np.array(cal.n_views)
        image_sizes_seen.add(cal.image_size)
    if "image_size" in existing:
        prev = tuple(int(v) for v in existing["image_size"])
        image_sizes_seen.add(prev)  # type: ignore[arg-type]
    if len(image_sizes_seen) > 1:
        print(
            f"Warning: mixed per-lens image sizes {image_sizes_seen}; "
            "keeping the most recent. Downstream rectify assumes a single size.",
            file=sys.stderr,
        )
    out["image_size"] = np.array(next(iter(image_sizes_seen)), dtype=np.int32)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(output), **out)


def main(args: Args) -> None:
    if args.front_video is None and args.back_video is None:
        print("Error: at least one of --front-video / --back-video is required", file=sys.stderr)
        sys.exit(1)

    if args.equidistant_only:
        existing: dict[str, np.ndarray] = {}
        if args.output.exists():
            existing = dict(np.load(str(args.output)))
            print(f"Loaded existing {args.output} -> will preserve lenses we don't (re)write")

        out: dict[str, np.ndarray] = dict(existing)
        image_size: tuple[int, int] | None = None
        for label, video in [("front", args.front_video), ("back", args.back_video)]:
            if video is None:
                continue
            size = _peek_image_size(video)
            K, D = _equidistant_intrinsics(size)
            out[f"K_{label}"] = K
            out[f"D_{label}"] = D
            out[f"rms_{label}"] = np.array(float("nan"))
            out[f"n_{label}"] = np.array(0)
            image_size = size
            print(
                f"[{label}] equidistant defaults from {size[0]}x{size[1]}: "
                f"f={K[0, 0]:.1f}  cx={K[0, 2]:.1f} cy={K[1, 2]:.1f}  D=[0, 0, 0, 0]"
            )
        if image_size is None:
            print("Error: no video given to read image size from", file=sys.stderr)
            sys.exit(1)
        out["image_size"] = np.array(image_size, dtype=np.int32)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(args.output), **out)
        print(
            f"\nWrote placeholder {args.output} — re-run without "
            f"--equidistant-only once you have a proper calibration recording."
        )
        return

    try:
        extract = build_extractor(args.target_type, args.marker_config, args.min_corners_per_view)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    results: dict[str, LensCalibration] = {}
    for label, video in [("front", args.front_video), ("back", args.back_video)]:
        if video is None:
            continue
        cal = calibrate_video(
            video, label, extract, args.frame_stride, args.min_spread_frac, args.min_views,
            fix_K=args.fix_K,
        )
        if cal is not None:
            results[label] = cal

    if not results:
        print("Error: no lens met --min-views; nothing saved", file=sys.stderr)
        sys.exit(1)

    save_intrinsics(args.output, results)
    keys = ", ".join(sorted(f"K_{k}" for k in results))
    print(f"\nWrote {args.output}  (calibrated: {keys})")


if __name__ == "__main__":
    main(tyro.cli(Args))
