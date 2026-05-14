#!/usr/bin/env python3
r"""Eye-to-hand calibration: estimate T_base_camera from ArUco marker images.

Loads images and joint states produced by ``capture_zed.py`` (with
``--enable-robot``).  For each frame the pipeline:

1. Detects all configured ArUco markers and runs ``solvePnP`` → ``T_camera_marker_i``
   for each detected marker
2. Loads the sidecar joint-state file and computes ``T_base_ee_i`` via MuJoCo FK
3. Chains: ``T_base_camera_i = T_base_ee_i @ T_ee_marker_k @ inv(T_camera_marker_i)``
   for each detected marker k

Outlier (image × marker) pairs are rejected, then ``T_base_camera`` is refined by
jointly minimizing reprojection error across all inlier observations and all markers.

Usage::

    python -m avantbot.perception.calibration.compute_pose \
        --image-dir calibration_images_20260310_120000

    python -m avantbot.perception.calibration.compute_pose \
        --image-dir ./imgs --output pose.npz
"""

from __future__ import annotations

import dataclasses
import shutil
import sys
from pathlib import Path  # noqa: TC003 – tyro needs Path at runtime

import cv2
import mujoco
import numpy as np
import pyzed.sl as sl
import tyro
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from avantbot.models import resolve_model
from avantbot.perception.calibration.capture_zed import RESOLUTION_MAP
from avantbot.perception.calibration.marker import (
    DEFAULT_MARKER_CONFIG,
    charuco_object_points,
    create_charuco_board,
    detect_aruco_markers,
    detect_charuco_corners,
    load_charuco_board_configs,
    load_marker_configs,
    load_T_box_board,
    load_T_box_marker,
    load_T_ee_box,
    marker_object_points,
    save_T_box_board,
    save_T_ee_box,
)

_DEFAULT_MODEL = "fr3_robotiq"
# (Was "fr3_scene_robotiq" pre-v0.3.x — that registry entry got dropped
# in 00b2874 when scenes moved to the runtime composer. The kinematic
# fragment is sufficient for FK / arm-base-to-EE pose computation.)


@dataclasses.dataclass
class Args:
    """Eye-to-hand calibration: T_base_camera from ArUco markers + robot FK."""

    image_dir: Path
    """Directory containing capture_XXXX.png and capture_XXXX_joints.npy files."""

    serial: str | None = None
    """ZED serial number for intrinsics. None picks the first available."""

    resolution: str = "HD720"
    """Resolution used during capture (must match for correct intrinsics)."""

    model_path: str = _DEFAULT_MODEL
    """MuJoCo model (registry name or path) for forward kinematics."""

    ee_site_name: str = "attachment_site"
    """MuJoCo site name on the end-effector for FK."""

    reproj_thresh: float = 1.0
    """Maximum mean reprojection error (px) to accept a frame as inlier."""

    refine_iters: int = 5
    """Max iterations of optimize → re-filter → re-optimize refinement."""

    fk_reg_weight: float = 20.0
    """L2 regularization weight (px per m or rad) on T_fk_correction parameters.
    Prevents the gauge degeneracy where T_fk_correction absorbs the camera pose.
    Increase to restrict the correction; set to 0 to disable."""

    bb_reg_weight: float = 50.0
    """L2 regularization weight on T_box_board deviation from its config value.
    Prevents the gauge degeneracy where T_ee_box and T_box_board drift by equal
    and opposite amounts. Increase to restrict board pose correction; set to 0
    to optimize freely (not recommended without a good initial estimate)."""

    marker_config: Path = DEFAULT_MARKER_CONFIG
    """Path to marker presets YAML file."""

    use_charuco: bool = False
    """Use ChArUco board pose estimation instead of single ArUco markers."""

    output: Path | None = None
    """Save optimized pose as .npz (T_base_camera, intrinsics, per-frame data)."""


# ---------------------------------------------------------------------------
# Intrinsics
# ---------------------------------------------------------------------------


def _load_intrinsics_from_file(
    image_dir: Path,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load intrinsics saved by ``capture_zed.py``, if available."""
    path = image_dir / "camera_intrinsics.npz"
    if not path.exists():
        return None
    data = np.load(str(path))
    K = data["camera_matrix"]
    dist = data["dist_coeffs"]
    res = str(data["resolution"]) if "resolution" in data else "unknown"
    print(f"Intrinsics loaded from {path}  (resolution={res}):")
    print(f"  fx={K[0, 0]:.1f}  fy={K[1, 1]:.1f}  cx={K[0, 2]:.1f}  cy={K[1, 2]:.1f}")
    print(f"  distortion coeffs: {dist}")
    return K, dist


def _load_zed_intrinsics(
    serial: str | None,
    resolution: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Open a ZED camera briefly to read factory-calibrated intrinsics.

    Returns (K, dist) where K is (3, 3) and dist is (N,) for the left camera.
    """
    cam = sl.Camera()
    init_params = sl.InitParameters()
    if serial is not None:
        init_params.set_from_serial_number(int(serial))
    else:
        devices = sl.Camera.get_device_list()
        if not devices:
            raise RuntimeError("No ZED cameras found")
        serial = str(devices[0].serial_number)
        init_params.set_from_serial_number(int(serial))
        print(f"Auto-selected ZED {serial}")

    init_params.camera_resolution = RESOLUTION_MAP[resolution]
    init_params.depth_mode = sl.DEPTH_MODE.NONE

    err = cam.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"ZED open failed: {err!r}")

    calib = cam.get_camera_information().camera_configuration.calibration_parameters
    left = calib.left_cam
    K = np.array([[left.fx, 0, left.cx], [0, left.fy, left.cy], [0, 0, 1]])
    dist = np.array(left.disto, dtype=np.float64)
    cam.close()

    print(f"Intrinsics loaded from camera (resolution={resolution}):")
    print(f"  fx={left.fx:.1f}  fy={left.fy:.1f}  cx={left.cx:.1f}  cy={left.cy:.1f}")
    print(f"  distortion coeffs: {dist}")
    return K, dist


# ---------------------------------------------------------------------------
# Forward kinematics via MuJoCo
# ---------------------------------------------------------------------------


def _compute_T_base_ee(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_positions: np.ndarray,
    ee_site_name: str,
) -> np.ndarray:
    """Forward kinematics: joint positions -> T_base_ee (4x4)."""
    data.qpos[: len(joint_positions)] = joint_positions
    mujoco.mj_forward(model, data)
    site_id = model.site(ee_site_name).id
    T = np.eye(4)
    T[:3, :3] = data.site_xmat[site_id].reshape(3, 3)
    T[:3, 3] = data.site_xpos[site_id]
    return T


# ---------------------------------------------------------------------------
# Pose helpers
# ---------------------------------------------------------------------------


def _rvec_tvec_to_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Convert Rodrigues rotation + translation to a 4x4 homogeneous matrix."""
    R = Rotation.from_rotvec(rvec).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec
    return T


def _T_to_rvec_tvec(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract rotvec and translation from a 4x4 homogeneous matrix."""
    rvec = Rotation.from_matrix(T[:3, :3]).as_rotvec()
    return rvec, T[:3, 3].copy()


# ---------------------------------------------------------------------------
# Per-frame estimation
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _FrameResult:
    filepath: Path
    marker_id: int  # ArUco ID, or -1 for ChArUco frames
    corners_2d: np.ndarray  # (N, 2) detected corners (4 for ArUco, N for ChArUco)
    joint_positions: np.ndarray  # (7,)
    T_camera_marker: np.ndarray  # (4, 4) from PnP
    T_base_ee: np.ndarray  # (4, 4) from FK
    T_base_camera: np.ndarray  # (4, 4) chained estimate
    reproj_error: float  # PnP reprojection error
    board_name: str | None = None  # ChArUco board name, or None for ArUco
    charuco_ids: np.ndarray | None = None  # (N, 1) int32 detected ChArUco corner IDs


def _estimate_poses(
    image_dir: Path,
    T_ee_markers: dict[int, np.ndarray],
    obj_pts_by_id: dict[int, np.ndarray],
    K: np.ndarray,
    dist: np.ndarray,
    mj_model: mujoco.MjModel | None = None,
    mj_data: mujoco.MjData | None = None,
    ee_site_name: str = "attachment_site",
) -> list[_FrameResult]:
    """Detect all configured markers, run PnP, get T_base_ee, and chain T_base_camera.

    Produces one ``_FrameResult`` per (image, detected marker) pair.
    Prefers ``capture_XXXX_ee_pose.npy`` (Franka calibrated FK) when available,
    falling back to MuJoCo FK from ``capture_XXXX_joints.npy``.
    """
    files = sorted(image_dir.glob("capture_*.png"))
    if not files:
        raise FileNotFoundError(f"No capture_*.png files found in {image_dir}")

    allowed_ids = set(T_ee_markers.keys())
    print(
        f"\nProcessing {len(files)} images from {image_dir} (marker IDs: {sorted(allowed_ids)}) ..."
    )
    results: list[_FrameResult] = []
    n_mujoco_fk = 0
    n_franka_tf = 0

    for filepath in files:
        stem = filepath.stem

        joints_path = filepath.parent / f"{stem}_joints.npy"
        ee_pose_path = filepath.parent / f"{stem}_ee_pose.npy"

        if not joints_path.exists() and not ee_pose_path.exists():
            print(f"  SKIP {filepath.name}: no joints or ee_pose file")
            continue

        img_bgr = cv2.imread(str(filepath))
        if img_bgr is None:
            print(f"  SKIP {filepath.name}: could not read image")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        corners, ids = detect_aruco_markers(img_rgb, allowed_ids=allowed_ids)
        if ids is None:
            print(f"  SKIP {filepath.name}: no configured markers detected")
            continue

        # Prefer Franka TF ee_pose (per-robot calibrated FK) when available.
        # Fall back to MuJoCo FK from joint angles.
        if ee_pose_path.exists():
            T_base_ee = np.load(str(ee_pose_path))
            joint_positions = np.load(str(joints_path)) if joints_path.exists() else np.zeros(7)
            n_franka_tf += 1
        else:
            joint_positions = np.load(str(joints_path))
            T_base_ee = _compute_T_base_ee(mj_model, mj_data, joint_positions, ee_site_name)
            n_mujoco_fk += 1

        for corn, mid in zip(corners, ids.flatten()):
            mid = int(mid)
            img_corners = corn.reshape(4, 2).astype(np.float64)
            obj_pts = obj_pts_by_id[mid]

            success, rvec, tvec = cv2.solvePnP(
                obj_pts,
                img_corners,
                K,
                dist,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if not success:
                print(f"  SKIP {filepath.name} marker={mid}: solvePnP failed")
                continue

            projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
            reproj_err = float(
                np.mean(np.linalg.norm(projected.reshape(4, 2) - img_corners, axis=1))
            )

            T_camera_marker = _rvec_tvec_to_T(rvec.flatten(), tvec.flatten())
            T_base_camera = T_base_ee @ T_ee_markers[mid] @ np.linalg.inv(T_camera_marker)

            results.append(
                _FrameResult(
                    filepath=filepath,
                    marker_id=mid,
                    corners_2d=img_corners,
                    joint_positions=joint_positions,
                    T_camera_marker=T_camera_marker,
                    T_base_ee=T_base_ee.copy(),
                    T_base_camera=T_base_camera,
                    reproj_error=reproj_err,
                )
            )
            bc_t = T_base_camera[:3, 3]
            print(
                f"  OK   {filepath.name} marker={mid}: reproj={reproj_err:.3f} px"
                f"  T_base_cam t=[{bc_t[0]:.4f}, {bc_t[1]:.4f}, {bc_t[2]:.4f}]"
            )

    print(
        f"  FK source: Franka TF ee_pose ({n_franka_tf} frames), MuJoCo fallback ({n_mujoco_fk} frames)"
    )

    return results


# ---------------------------------------------------------------------------
# ChArUco per-frame estimation
# ---------------------------------------------------------------------------


def _estimate_poses_charuco(
    image_dir: Path,
    board_name: str,
    board: "cv2.aruco.CharucoBoard",
    T_ee_box: np.ndarray,
    T_box_board: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    mj_model: "mujoco.MjModel | None" = None,
    mj_data: "mujoco.MjData | None" = None,
    ee_site_name: str = "attachment_site",
) -> list[_FrameResult]:
    """Detect ChArUco corners, run PnP, get T_base_ee, and chain T_base_camera.

    Produces one ``_FrameResult`` per image where enough corners are detected.
    Prefers ``capture_XXXX_ee_pose.npy`` (Franka calibrated FK) when available,
    falling back to MuJoCo FK from ``capture_XXXX_joints.npy``.
    """
    files = sorted(image_dir.glob("capture_*.png"))
    if not files:
        raise FileNotFoundError(f"No capture_*.png files found in {image_dir}")

    T_ee_board = T_ee_box @ T_box_board
    print(f"\nProcessing {len(files)} images from {image_dir} (ChArUco board: '{board_name}') ...")
    results: list[_FrameResult] = []
    n_mujoco_fk = 0
    n_franka_tf = 0

    for filepath in files:
        stem = filepath.stem

        joints_path = filepath.parent / f"{stem}_joints.npy"
        ee_pose_path = filepath.parent / f"{stem}_ee_pose.npy"

        if not joints_path.exists() and not ee_pose_path.exists():
            print(f"  SKIP {filepath.name}: no joints or ee_pose file")
            continue

        img_bgr = cv2.imread(str(filepath))
        if img_bgr is None:
            print(f"  SKIP {filepath.name}: could not read image")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        ch_corners, ch_ids, _, _ = detect_charuco_corners(img_rgb, board, K, dist)
        if ch_ids is None:
            print(f"  SKIP {filepath.name}: not enough ChArUco corners detected")
            continue

        if ee_pose_path.exists():
            T_base_ee = np.load(str(ee_pose_path))
            joint_positions = np.load(str(joints_path)) if joints_path.exists() else np.zeros(7)
            n_franka_tf += 1
        else:
            joint_positions = np.load(str(joints_path))
            T_base_ee = _compute_T_base_ee(mj_model, mj_data, joint_positions, ee_site_name)
            n_mujoco_fk += 1

        obj_pts = charuco_object_points(board, ch_ids)
        img_pts = ch_corners.reshape(-1, 2).astype(np.float64)
        n_corners = len(ch_ids)

        success, rvec, tvec = cv2.solvePnP(
            obj_pts,
            img_pts,
            K,
            dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            print(f"  SKIP {filepath.name}: solvePnP failed")
            continue

        projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
        reproj_err = float(
            np.mean(np.linalg.norm(projected.reshape(n_corners, 2) - img_pts, axis=1))
        )

        T_camera_board = _rvec_tvec_to_T(rvec.flatten(), tvec.flatten())
        T_base_camera = T_base_ee @ T_ee_board @ np.linalg.inv(T_camera_board)

        # Chain consistency check: re-project via T_base_camera chain.
        T_camera_base = np.linalg.inv(T_base_camera)
        T_cam_board_chain = T_camera_base @ T_base_ee @ T_ee_board
        rvec_c, tvec_c = _T_to_rvec_tvec(T_cam_board_chain)
        proj_chain, _ = cv2.projectPoints(obj_pts, rvec_c, tvec_c, K, dist)
        chain_err = float(
            np.mean(np.linalg.norm(proj_chain.reshape(n_corners, 2) - img_pts, axis=1))
        )
        if chain_err > 5.0:
            print(
                f"  WARN {filepath.name}: chain reproj={chain_err:.1f} px (PnP={reproj_err:.2f} px)"
                "  → T_box_board may be wrong"
            )

        results.append(
            _FrameResult(
                filepath=filepath,
                marker_id=-1,
                corners_2d=img_pts,
                joint_positions=joint_positions,
                T_camera_marker=T_camera_board,
                T_base_ee=T_base_ee.copy(),
                T_base_camera=T_base_camera,
                reproj_error=reproj_err,
                board_name=board_name,
                charuco_ids=ch_ids.copy(),
            )
        )
        bc_t = T_base_camera[:3, 3]
        print(
            f"  OK   {filepath.name} charuco n={n_corners}: reproj={reproj_err:.3f} px"
            f"  T_base_cam t=[{bc_t[0]:.4f}, {bc_t[1]:.4f}, {bc_t[2]:.4f}]"
        )

    print(
        f"  FK source: Franka TF ee_pose ({n_franka_tf} frames), MuJoCo fallback ({n_mujoco_fk} frames)"
    )
    return results


def _update_frame_estimates_charuco(
    results: list[_FrameResult],
    T_ee_box: np.ndarray,
    T_box_board: np.ndarray,
    T_fk_correction: np.ndarray,
) -> None:
    """Recompute T_base_camera on each ChArUco frame using refined transforms."""
    T_ee_board = T_ee_box @ T_box_board
    for r in results:
        r.T_base_camera = (
            T_fk_correction @ r.T_base_ee @ T_ee_board @ np.linalg.inv(r.T_camera_marker)
        )


def _reprojection_residuals_charuco(
    params: np.ndarray,
    board: "cv2.aruco.CharucoBoard",
    results: list[_FrameResult],
    K: np.ndarray,
    dist: np.ndarray,
    fk_reg_weight: float = 0.0,
    bb_reg_weight: float = 0.0,
    bb_ref: np.ndarray | None = None,
) -> np.ndarray:
    """Residuals for jointly optimizing T_base_camera, T_ee_box, T_fk_correction, T_box_board.

    ``params`` is (24,): [rvec_bc(3), tvec_bc(3), rvec_eb(3), tvec_eb(3),
                          rvec_fc(3), tvec_fc(3), rvec_bb(3), tvec_bb(3)]

    ``bb_ref`` is the (6,) reference rotvec+tvec for T_box_board regularization
    (deviation from config value, not from zero).
    """
    T_base_camera = _rvec_tvec_to_T(params[:3], params[3:6])
    T_ee_box = _rvec_tvec_to_T(params[6:9], params[9:12])
    T_fk_correction = _rvec_tvec_to_T(params[12:15], params[15:18])
    T_box_board = _rvec_tvec_to_T(params[18:21], params[21:24])
    T_camera_base = np.linalg.inv(T_base_camera)
    T_ee_board = T_ee_box @ T_box_board

    per_frame: list[np.ndarray] = []
    for r in results:
        assert r.charuco_ids is not None
        obj_pts = charuco_object_points(board, r.charuco_ids)
        T_base_ee_corrected = T_fk_correction @ r.T_base_ee
        T_camera_board_i = T_camera_base @ T_base_ee_corrected @ T_ee_board
        rvec_i, tvec_i = _T_to_rvec_tvec(T_camera_board_i)
        n = len(r.corners_2d)
        projected, _ = cv2.projectPoints(obj_pts, rvec_i, tvec_i, K, dist)
        per_frame.append((r.corners_2d - projected.reshape(n, 2)).ravel())

    flat = np.concatenate(per_frame)
    regs = []
    if fk_reg_weight > 0.0:
        regs.append(fk_reg_weight * params[12:18])
    if bb_reg_weight > 0.0 and bb_ref is not None:
        regs.append(bb_reg_weight * (params[18:24] - bb_ref))
    if regs:
        return np.concatenate([flat, *regs])
    return flat


def _optimize_T_base_camera_charuco(
    inliers: list[_FrameResult],
    board: "cv2.aruco.CharucoBoard",
    T_ee_box: np.ndarray,
    T_fk_correction: np.ndarray,
    T_box_board: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    fk_reg_weight: float = 0.0,
    bb_reg_weight: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Jointly refine T_base_camera, T_ee_box, T_fk_correction, T_box_board for ChArUco frames.

    Returns (T_base_camera, T_ee_box, T_fk_correction, T_box_board, mean_reproj_error).
    """
    bc_rvecs = np.array(
        [Rotation.from_matrix(r.T_base_camera[:3, :3]).as_rotvec() for r in inliers]
    )
    bc_tvecs = np.array([r.T_base_camera[:3, 3] for r in inliers])

    r_bc_init = np.median(bc_rvecs, axis=0)
    t_bc_init = np.median(bc_tvecs, axis=0)
    r_eb_init, t_eb_init = _T_to_rvec_tvec(T_ee_box)
    r_fc_init, t_fc_init = _T_to_rvec_tvec(T_fk_correction)
    r_bb_init, t_bb_init = _T_to_rvec_tvec(T_box_board)
    bb_ref = np.concatenate([r_bb_init, t_bb_init])  # anchor for regularization

    x0 = np.concatenate(
        [
            r_bc_init,
            t_bc_init,
            r_eb_init,
            t_eb_init,
            r_fc_init,
            t_fc_init,
            r_bb_init,
            t_bb_init,
        ]
    )

    total_corners = sum(len(r.corners_2d) for r in inliers)

    # Diagnostic: reprojection error at x0 (before optimization).
    r0 = _reprojection_residuals_charuco(
        x0, board, inliers, K, dist, fk_reg_weight, bb_reg_weight, bb_ref
    )
    r0_px = r0[: total_corners * 2].reshape(-1, 2)
    print(f"    Init reproj error (x0): {np.mean(np.linalg.norm(r0_px, axis=1)):.4f} px")

    result = least_squares(
        _reprojection_residuals_charuco,
        x0,
        args=(board, inliers, K, dist, fk_reg_weight, bb_reg_weight, bb_ref),
        method="lm",
    )

    T_bc_opt = _rvec_tvec_to_T(result.x[:3], result.x[3:6])
    T_ee_box_opt = _rvec_tvec_to_T(result.x[6:9], result.x[9:12])
    T_fk_correction_opt = _rvec_tvec_to_T(result.x[12:15], result.x[15:18])
    T_box_board_opt = _rvec_tvec_to_T(result.x[18:21], result.x[21:24])

    pixel_residuals = result.fun[: total_corners * 2].reshape(-1, 2)
    mean_err = float(np.mean(np.linalg.norm(pixel_residuals, axis=1)))

    return T_bc_opt, T_ee_box_opt, T_fk_correction_opt, T_box_board_opt, mean_err


# ---------------------------------------------------------------------------
# Update per-frame estimates with a new T_ee_marker
# ---------------------------------------------------------------------------


def _update_frame_estimates(
    results: list[_FrameResult],
    T_ee_box: np.ndarray,
    T_box_markers: dict[int, np.ndarray],
    T_fk_correction: np.ndarray,
) -> None:
    """Recompute T_base_camera on each frame using refined transforms."""
    for r in results:
        T_ee_marker = T_ee_box @ T_box_markers[r.marker_id]
        r.T_base_camera = (
            T_fk_correction @ r.T_base_ee @ T_ee_marker @ np.linalg.inv(r.T_camera_marker)
        )


# ---------------------------------------------------------------------------
# Outlier rejection
# ---------------------------------------------------------------------------


def _reject_outliers(
    results: list[_FrameResult],
    reproj_thresh: float,
    sigma_factor: float = 2.0,
) -> list[_FrameResult]:
    """Two-stage outlier rejection on T_base_camera estimates."""
    stage1 = [r for r in results if r.reproj_error <= reproj_thresh]
    n_reproj_rejected = len(results) - len(stage1)
    if n_reproj_rejected:
        print(f"\nStage 1: rejected {n_reproj_rejected} frames with reproj > {reproj_thresh} px")

    if len(stage1) < 3:
        print(
            f"Warning: only {len(stage1)} frames after reproj filter, skipping statistical filter"
        )
        return stage1

    rvecs = np.array([Rotation.from_matrix(r.T_base_camera[:3, :3]).as_rotvec() for r in stage1])
    tvecs = np.array([r.T_base_camera[:3, 3] for r in stage1])

    r_median = np.median(rvecs, axis=0)
    t_median = np.median(tvecs, axis=0)
    r_std = np.maximum(np.std(rvecs, axis=0), 1e-8)
    t_std = np.maximum(np.std(tvecs, axis=0), 1e-8)

    stage2: list[_FrameResult] = []
    for i, r in enumerate(stage1):
        r_dev = np.abs(rvecs[i] - r_median) / r_std
        t_dev = np.abs(tvecs[i] - t_median) / t_std
        if np.all(r_dev < sigma_factor) and np.all(t_dev < sigma_factor):
            stage2.append(r)

    n_stat_rejected = len(stage1) - len(stage2)
    if n_stat_rejected:
        print(f"Stage 2: rejected {n_stat_rejected} statistical outlier frames")

    print(f"Inliers: {len(stage2)} / {len(results)} frames")
    return stage2


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------


def _reprojection_residuals(
    params: np.ndarray,
    T_box_markers: dict[int, np.ndarray],
    obj_pts_by_id: dict[int, np.ndarray],
    all_corners_2d: np.ndarray,
    all_T_base_ee: np.ndarray,
    all_marker_ids: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    fk_reg_weight: float = 0.0,
) -> np.ndarray:
    """Residuals for jointly optimizing T_base_camera, T_ee_box, and T_fk_correction.

    ``params`` is (18,): [rvec_bc(3), tvec_bc(3), rvec_eb(3), tvec_eb(3), rvec_fc(3), tvec_fc(3)]

    For each (frame, marker) result i::

        T_base_ee_corrected = T_fk_correction @ T_base_ee_i
        T_ee_marker_k = T_ee_box @ T_box_markers[k]
        T_camera_marker_i = inv(T_base_camera) @ T_base_ee_corrected @ T_ee_marker_k

    Returns flattened residuals (N*4*2 [+ 6],).
    """
    T_base_camera = _rvec_tvec_to_T(params[:3], params[3:6])
    T_ee_box = _rvec_tvec_to_T(params[6:9], params[9:12])
    T_fk_correction = _rvec_tvec_to_T(params[12:15], params[15:18])
    T_camera_base = np.linalg.inv(T_base_camera)

    n_results = all_corners_2d.shape[0]
    residuals = np.empty((n_results, 4, 2))

    for i in range(n_results):
        mid = int(all_marker_ids[i])
        T_ee_marker = T_ee_box @ T_box_markers[mid]
        T_base_ee_corrected = T_fk_correction @ all_T_base_ee[i]
        T_camera_marker_i = T_camera_base @ T_base_ee_corrected @ T_ee_marker
        rvec_i, tvec_i = _T_to_rvec_tvec(T_camera_marker_i)
        projected, _ = cv2.projectPoints(obj_pts_by_id[mid], rvec_i, tvec_i, K, dist)
        residuals[i] = all_corners_2d[i] - projected.reshape(4, 2)

    flat = residuals.ravel()
    if fk_reg_weight > 0.0:
        return np.concatenate([flat, fk_reg_weight * params[12:18]])
    return flat


def _optimize_T_base_camera(
    inliers: list[_FrameResult],
    T_ee_box: np.ndarray,
    T_fk_correction: np.ndarray,
    T_box_markers: dict[int, np.ndarray],
    obj_pts_by_id: dict[int, np.ndarray],
    K: np.ndarray,
    dist: np.ndarray,
    fk_reg_weight: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Jointly refine T_base_camera, T_ee_box, and T_fk_correction by minimizing reprojection error.

    T_box_markers is held fixed. Returns (T_base_camera, T_ee_box, T_fk_correction, mean_reproj_error).
    """
    bc_rvecs = np.array(
        [Rotation.from_matrix(r.T_base_camera[:3, :3]).as_rotvec() for r in inliers]
    )
    bc_tvecs = np.array([r.T_base_camera[:3, 3] for r in inliers])

    r_bc_init = np.median(bc_rvecs, axis=0)
    t_bc_init = np.median(bc_tvecs, axis=0)
    r_eb_init, t_eb_init = _T_to_rvec_tvec(T_ee_box)
    r_fc_init, t_fc_init = _T_to_rvec_tvec(T_fk_correction)

    x0 = np.concatenate([r_bc_init, t_bc_init, r_eb_init, t_eb_init, r_fc_init, t_fc_init])

    all_corners = np.array([r.corners_2d for r in inliers])  # (N, 4, 2)
    all_T_base_ee = np.array([r.T_base_ee for r in inliers])  # (N, 4, 4)
    all_marker_ids = np.array([r.marker_id for r in inliers])  # (N,)

    result = least_squares(
        _reprojection_residuals,
        x0,
        args=(
            T_box_markers,
            obj_pts_by_id,
            all_corners,
            all_T_base_ee,
            all_marker_ids,
            K,
            dist,
            fk_reg_weight,
        ),
        method="lm",
    )

    T_bc_opt = _rvec_tvec_to_T(result.x[:3], result.x[3:6])
    T_ee_box_opt = _rvec_tvec_to_T(result.x[6:9], result.x[9:12])
    T_fk_correction_opt = _rvec_tvec_to_T(result.x[12:15], result.x[15:18])

    # Mean reprojection error from pixel residuals only (exclude regularization terms).
    n_pixel_residuals = len(inliers) * 4 * 2
    pixel_residuals = result.fun[:n_pixel_residuals].reshape(-1, 4, 2)
    per_frame_err = np.mean(np.linalg.norm(pixel_residuals, axis=2), axis=1)
    mean_err = float(np.mean(per_frame_err))

    return T_bc_opt, T_ee_box_opt, T_fk_correction_opt, mean_err


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: Args) -> None:
    """Run the eye-to-hand calibration pipeline."""
    if not args.image_dir.is_dir():
        print(f"Error: {args.image_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    cached = _load_intrinsics_from_file(args.image_dir)
    if cached is not None:
        K, dist = cached
    else:
        K, dist = _load_zed_intrinsics(args.serial, args.resolution)

    resolved_model = str(resolve_model(args.model_path))
    mj_model = mujoco.MjModel.from_xml_path(resolved_model)
    mj_data = mujoco.MjData(mj_model)
    print(f"MuJoCo model loaded: {resolved_model}  (ee_site={args.ee_site_name})")

    T_ee_box = load_T_ee_box(args.marker_config)

    if args.use_charuco:
        charuco_board_configs = load_charuco_board_configs(args.marker_config)
        if not charuco_board_configs:
            print(f"Error: no charuco boards defined in {args.marker_config}", file=sys.stderr)
            sys.exit(1)
        board_name, board_cfg = next(iter(charuco_board_configs.items()))
        charuco_board = create_charuco_board(board_cfg)
        T_box_board = load_T_box_board(args.marker_config, board_name)
        print(
            f"ChArUco board '{board_name}': {board_cfg.squares_x}x{board_cfg.squares_y} "
            f"squares, square={board_cfg.square_length * 1000:.0f}mm, "
            f"marker={board_cfg.marker_length * 1000:.0f}mm"
        )

        results = _estimate_poses_charuco(
            args.image_dir,
            board_name,
            charuco_board,
            T_ee_box,
            T_box_board,
            K,
            dist,
            mj_model,
            mj_data,
            args.ee_site_name,
        )
        if not results:
            print("Error: no ChArUco results found", file=sys.stderr)
            sys.exit(1)
        print(f"\nTotal ChArUco frame results: {len(results)}")

        T_ee_box_cur = T_ee_box.copy()
        T_box_board_cur = T_box_board.copy()
        T_fk_correction_cur = np.eye(4)
        best_err = float("inf")
        best_result: tuple | None = None

        for iteration in range(1, args.refine_iters + 1):
            inliers = _reject_outliers(results, args.reproj_thresh)
            if not inliers:
                print("Error: all frames rejected as outliers", file=sys.stderr)
                sys.exit(1)

            print(
                f"\n[Iter {iteration}/{args.refine_iters}] "
                f"Optimizing over {len(inliers)} inlier ChArUco frames ..."
            )

            T_base_camera, T_ee_box_opt, T_fk_correction_opt, T_box_board_opt, mean_err = (
                _optimize_T_base_camera_charuco(
                    inliers,
                    charuco_board,
                    T_ee_box_cur,
                    T_fk_correction_cur,
                    T_box_board_cur,
                    K,
                    dist,
                    fk_reg_weight=args.fk_reg_weight,
                    bb_reg_weight=args.bb_reg_weight,
                )
            )

            improvement = best_err - mean_err
            print(f"  Mean reproj error: {mean_err:.4f} px  (delta={improvement:+.4f} px)")

            if mean_err < best_err:
                best_err = mean_err
                best_result = (T_base_camera, T_ee_box_opt, T_fk_correction_opt, T_box_board_opt)

            if improvement > 0.01 and iteration < args.refine_iters:
                T_ee_box_cur = T_ee_box_opt
                T_box_board_cur = T_box_board_opt
                T_fk_correction_cur = T_fk_correction_opt
                _update_frame_estimates_charuco(
                    results, T_ee_box_cur, T_box_board_cur, T_fk_correction_cur
                )
            else:
                if improvement < 0:
                    print(
                        f"  Warning: optimization diverged — keeping best result ({best_err:.4f} px)"
                    )
                elif improvement <= 0.01:
                    print("  Converged (improvement < 0.01 px)")
                break

    else:
        marker_configs = load_marker_configs(args.marker_config)
        if not marker_configs:
            print(f"Error: no markers defined in {args.marker_config}", file=sys.stderr)
            sys.exit(1)

        T_box_markers: dict[int, np.ndarray] = {}
        obj_pts_by_id: dict[int, np.ndarray] = {}

        print(f"Loaded {len(marker_configs)} marker configs from {args.marker_config}:")
        for name, cfg in marker_configs.items():
            T_box_markers[cfg.id] = load_T_box_marker(args.marker_config, cfg.id)
            obj_pts_by_id[cfg.id] = marker_object_points(cfg.size)
            print(f"  '{name}' id={cfg.id} size={cfg.size} m")

        T_ee_markers = {mid: T_ee_box @ T_bm for mid, T_bm in T_box_markers.items()}

        results = _estimate_poses(
            args.image_dir,
            T_ee_markers,
            obj_pts_by_id,
            K,
            dist,
            mj_model,
            mj_data,
            args.ee_site_name,
        )
        if not results:
            print("Error: no (image × marker) results found", file=sys.stderr)
            sys.exit(1)
        print(f"\nTotal (image × marker) result pairs: {len(results)}")

        T_ee_box_cur = T_ee_box.copy()
        T_fk_correction_cur = np.eye(4)
        best_err = float("inf")
        best_result = None

        for iteration in range(1, args.refine_iters + 1):
            inliers = _reject_outliers(results, args.reproj_thresh)
            if not inliers:
                print("Error: all frames rejected as outliers", file=sys.stderr)
                sys.exit(1)

            print(
                f"\n[Iter {iteration}/{args.refine_iters}] "
                f"Optimizing over {len(inliers)} inlier (image × marker) pairs ..."
            )

            T_base_camera, T_ee_box_opt, T_fk_correction_opt, mean_err = _optimize_T_base_camera(
                inliers,
                T_ee_box_cur,
                T_fk_correction_cur,
                T_box_markers,
                obj_pts_by_id,
                K,
                dist,
                fk_reg_weight=args.fk_reg_weight,
            )

            improvement = best_err - mean_err
            print(f"  Mean reproj error: {mean_err:.4f} px  (delta={improvement:+.4f} px)")

            if mean_err < best_err:
                best_err = mean_err
                best_result = (T_base_camera, T_ee_box_opt, T_fk_correction_opt)

            if improvement > 0.01 and iteration < args.refine_iters:
                T_ee_box_cur = T_ee_box_opt
                T_fk_correction_cur = T_fk_correction_opt
                _update_frame_estimates(results, T_ee_box_cur, T_box_markers, T_fk_correction_cur)
            else:
                if improvement < 0:
                    print(
                        f"  Warning: optimization diverged — keeping best result ({best_err:.4f} px)"
                    )
                elif improvement <= 0.01:
                    print("  Converged (improvement < 0.01 px)")
                break

    assert best_result is not None
    if args.use_charuco:
        T_base_camera, T_ee_box_opt, T_fk_correction_opt, T_box_board_opt = best_result
    else:
        T_base_camera, T_ee_box_opt, T_fk_correction_opt = best_result
        T_box_board_opt = None
    mean_err = best_err

    rvec_opt, tvec_opt = _T_to_rvec_tvec(T_base_camera)
    rpy = Rotation.from_rotvec(rvec_opt).as_euler("xyz", degrees=True)

    rvec_eb, tvec_eb = _T_to_rvec_tvec(T_ee_box_opt)
    rpy_eb = Rotation.from_rotvec(rvec_eb).as_euler("xyz", degrees=True)
    dt_eb = T_ee_box_opt[:3, 3] - T_ee_box[:3, 3]
    rpy_eb_cfg = Rotation.from_matrix(T_ee_box[:3, :3]).as_euler("xyz", degrees=True)
    drpy_eb = rpy_eb - rpy_eb_cfg

    rvec_fc, tvec_fc = _T_to_rvec_tvec(T_fk_correction_opt)
    rpy_fc = Rotation.from_rotvec(rvec_fc).as_euler("xyz", degrees=True)

    print("\n" + "=" * 60)
    print("Optimized T_base_camera (4x4):")
    print(np.array2string(T_base_camera, precision=6, suppress_small=True))
    print(f"\nMean reprojection error: {mean_err:.4f} px")
    print(f"Translation (m): {tvec_opt}")
    print(f"Rotation (deg):  roll={rpy[0]:.2f}  pitch={rpy[1]:.2f}  yaw={rpy[2]:.2f}")

    print()
    print("Refined T_ee_box (4x4):")
    print(np.array2string(T_ee_box_opt, precision=6, suppress_small=True))
    print(f"Translation (m): {tvec_eb}")
    print(f"Rotation (deg):  roll={rpy_eb[0]:.2f}  pitch={rpy_eb[1]:.2f}  yaw={rpy_eb[2]:.2f}")
    print(
        f"delta from config:  dt(mm)=[{dt_eb[0] * 1e3:.1f}, {dt_eb[1] * 1e3:.1f}, {dt_eb[2] * 1e3:.1f}]"
        f"  drpy(deg)=[{drpy_eb[0]:.2f}, {drpy_eb[1]:.2f}, {drpy_eb[2]:.2f}]"
    )

    print()
    print("FK correction T_fk_correction (global MuJoCo URDF offset, should be near identity):")
    print(np.array2string(T_fk_correction_opt, precision=6, suppress_small=True))
    print(
        f"Translation (mm): [{tvec_fc[0] * 1e3:.1f}, {tvec_fc[1] * 1e3:.1f}, {tvec_fc[2] * 1e3:.1f}]"
        f"  Rotation (deg): roll={rpy_fc[0]:.2f}  pitch={rpy_fc[1]:.2f}  yaw={rpy_fc[2]:.2f}"
    )

    if T_box_board_opt is not None:
        rvec_bb, tvec_bb = _T_to_rvec_tvec(T_box_board_opt)
        rpy_bb = Rotation.from_rotvec(rvec_bb).as_euler("xyz", degrees=True)
        dt_bb = T_box_board_opt[:3, 3] - T_box_board[:3, 3]
        rpy_bb_cfg = Rotation.from_matrix(T_box_board[:3, :3]).as_euler("xyz", degrees=True)
        drpy_bb = rpy_bb - rpy_bb_cfg
        print()
        print("Refined T_box_board (4x4):")
        print(np.array2string(T_box_board_opt, precision=6, suppress_small=True))
        print(f"Translation (m): {tvec_bb}")
        print(f"Rotation (deg):  roll={rpy_bb[0]:.2f}  pitch={rpy_bb[1]:.2f}  yaw={rpy_bb[2]:.2f}")
        print(
            f"delta from config:  dt(mm)=[{dt_bb[0] * 1e3:.1f}, {dt_bb[1] * 1e3:.1f}, {dt_bb[2] * 1e3:.1f}]"
            f"  drpy(deg)=[{drpy_bb[0]:.2f}, {drpy_bb[1]:.2f}, {drpy_bb[2]:.2f}]"
        )

    print("=" * 60)

    # Write refined transforms to a new config YAML in the image dir (user-writable).
    config_stem = args.marker_config.stem
    output_config = args.image_dir / f"{config_stem}_computed_{args.image_dir.name}.yaml"
    shutil.copy(args.marker_config, output_config)
    save_T_ee_box(output_config, T_ee_box_opt)
    if T_box_board_opt is not None:
        save_T_box_board(output_config, board_name, T_box_board_opt)
    print(f"\nRefined config written to {output_config}")

    if args.output is not None:
        inlier_files = [str(r.filepath) for r in inliers]
        reproj_errors = [r.reproj_error for r in inliers]
        inlier_marker_ids = [r.marker_id for r in inliers]
        all_T_base_ee_arr = np.array([r.T_base_ee for r in inliers])

        savez_kwargs: dict = dict(
            T_base_camera=T_base_camera,
            T_ee_box_opt=T_ee_box_opt,
            T_ee_box_config=T_ee_box,
            T_fk_correction=T_fk_correction_opt,
            K=K,
            dist=dist,
            mean_reproj_error=mean_err,
            inlier_files=inlier_files,
            reproj_errors=reproj_errors,
            inlier_marker_ids=inlier_marker_ids,
            all_T_base_ee=all_T_base_ee_arr,
        )
        if T_box_board_opt is not None:
            savez_kwargs["T_box_board_opt"] = T_box_board_opt
            savez_kwargs["T_box_board_config"] = T_box_board
        np.savez(str(args.output), **savez_kwargs)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main(tyro.cli(Args))
