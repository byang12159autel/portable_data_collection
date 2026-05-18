"""Stage protocols + Pipeline composition for the marker-pose toolkit.

Five swappable stages, each declared as a ``Protocol``. Implementations
live in their respective packages:

    FrameSource      -- core.camera
    Calibrator       -- calibration
    Rectifier        -- core.rectify
    Detector         -- core.markers          (data-producing)
    DrawingDetector  -- core.markers          (visualization)
    PoseEstimator    -- pose_6d

Protocols are structural: any class whose methods match qualifies, no
inheritance required. To plug in a new calibration method, write a class
with ``__call__(source) -> dict`` and drop it under ``calibration/`` --
the rest of the chain is unchanged because every stage references the
next only through its protocol contract.

Two app shapes use these stages:

**Data pipeline** (rectify -> detect -> [optional pose]) — for apps
that produce numerical output (poses, corners, trajectories)::

    pipeline = DataPipeline(
        rectifier=my_rectifier,
        detector=my_detector,           # returns Detections
        pose_estimator=my_pose,         # optional
    )
    rectified, detections, pose = pipeline.process(frame)

**Preview pipeline** (rectify -> draw) — for visualization apps where
detection and overlay are folded together by ``core.markers``'s
``_build_detector_fns``::

    pipeline = PreviewPipeline(
        rectifier=my_rectifier,
        detect_fns=[fn1, fn2, ...],     # each returns (annotated_rgb, count)
    )
    annotated_rgb, total_count = pipeline.process_bgr(bgr_frame)
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Iterator, Protocol, runtime_checkable

import cv2
import numpy as np


@dataclasses.dataclass
class Detections:
    """Marker corners + IDs found in a single frame.

    ``corners[i]`` is the (4, 2) float32 pixel quad for the marker with
    ID ``ids[i]``. Empty arrays when nothing was detected.
    """

    corners: np.ndarray  # (N, 4, 2) float32
    ids: np.ndarray      # (N,) int32


@dataclasses.dataclass
class FramePose:
    """Camera pose for a single frame, or ``None`` if PnP failed."""

    T_world_camera: np.ndarray | None  # (4, 4) float64 or None
    n_inliers: int = 0


@runtime_checkable
class FrameSource(Protocol):
    """Yields BGR ndarrays. ``__len__`` may raise for unbounded live sources."""

    def __iter__(self) -> Iterator[np.ndarray]: ...
    def __len__(self) -> int: ...


@runtime_checkable
class Rectifier(Protocol):
    """Maps a raw camera frame to an undistorted pinhole frame.

    ``K_pinhole`` is the intrinsics of the output frame; downstream
    ``cv2.solvePnP`` uses it with ``distCoeffs=0``.
    """

    K_pinhole: np.ndarray

    def apply(self, frame: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class Detector(Protocol):
    """Finds markers in a rectified frame and returns the raw data."""

    def __call__(self, frame: np.ndarray) -> Detections: ...


DrawingDetector = Callable[[np.ndarray], "tuple[np.ndarray, int]"]
"""Detect + overlay in one call: ``(rgb) -> (annotated_rgb, count)``.

This is the shape produced by ``markers.detect._build_detector_fns`` and
consumed by the viser preview apps. Use it when you don't need the raw
corner data downstream.
"""


@runtime_checkable
class PoseEstimator(Protocol):
    """Camera pose from per-frame detections.

    Stateful estimators (e.g. learned-layout, which builds a marker map
    from co-visibility before tracking) own their own ``learn(frames)``
    pre-pass; that method is implementation-specific and not part of the
    protocol.
    """

    def __call__(self, detections: Detections, K: np.ndarray) -> FramePose: ...


@runtime_checkable
class Calibrator(Protocol):
    """Estimates camera intrinsics from a calibration recording.

    Return value is an npz-compatible dict keyed for ``np.savez``.
    """

    def __call__(self, source: FrameSource) -> dict: ...


@dataclasses.dataclass
class GripperState:
    """One frame's gripper-state estimate.

    Fields are populated only when the underlying step succeeded:

      - ``H_plane_to_img`` / ``H_img_to_plane``: set when the anchor marker
        was detected and single-marker PnP returned a valid pose.
      - ``centers_plane``: the 4 chopstick-dot positions in marker-plane
        metres, populated when dot detection found enough dots to compute
        an angle.
      - ``ordered_centers``: the same 4 points keyed as ``left_top``,
        ``left_bottom``, ``right_top``, ``right_bottom``.
      - ``angle_deg``: the in-plane chopstick hinge angle.

    Diagnostic counters (``n_white_dots``, ``n_black_dots``) carry the
    number of dot candidates that survived the marker-area filter, useful
    for tuning thresholds.
    """

    angle_deg: float | None = None
    centers_plane: np.ndarray | None = None
    ordered_centers: dict[str, np.ndarray] | None = None
    H_plane_to_img: np.ndarray | None = None
    H_img_to_plane: np.ndarray | None = None
    n_white_dots: int = 0
    n_black_dots: int = 0


@runtime_checkable
class GripperEstimator(Protocol):
    """Per-frame gripper-state estimator.

    Consumes the rectified frame (needs raw pixels for dot detection),
    the marker ``Detections`` from the shared ``Detector`` stage, and the
    rectified-camera intrinsics. Returns a ``GripperState``; on failure
    of any sub-step the relevant fields are simply left as ``None`` /
    ``0`` rather than raising, so the caller can decide what to log.

    Implementations own the choice of *which* marker anchors the hinge
    plane — they filter ``Detections`` for their configured marker ID.
    """

    def __call__(
        self,
        rectified: np.ndarray,
        detections: Detections,
        K: np.ndarray,
    ) -> GripperState: ...


@dataclasses.dataclass
class DataPipeline:
    """rectify -> detect -> [optional pose].

    Swap any stage by passing a different implementation to the
    constructor. Pose estimation is optional -- omit it for apps that
    only need detections (e.g. for benchmarking).
    """

    rectifier: Rectifier
    detector: Detector
    pose_estimator: PoseEstimator | None = None

    def process(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray, Detections, FramePose | None]:
        rectified = self.rectifier.apply(frame)
        detections = self.detector(rectified)
        if self.pose_estimator is None:
            return rectified, detections, None
        pose = self.pose_estimator(detections, self.rectifier.K_pinhole)
        return rectified, detections, pose


@dataclasses.dataclass
class PreviewPipeline:
    """rectify -> draw detections, for viser-style previewers.

    ``detect_fns`` are visualizing-detector callables -- each takes an
    RGB frame and returns ``(annotated_rgb, count)``. They run in order
    and the annotated frame from one feeds the next, so a "multi" config
    layers AprilGrid + ArUco overlays on the same frame.

    Use one of these per camera/lens; the calling app composes the
    rendered panels however it likes.
    """

    rectifier: Rectifier
    detect_fns: list[DrawingDetector]

    def process_bgr(self, bgr: np.ndarray) -> tuple[np.ndarray, int]:
        """Rectify the raw BGR frame, run all detect_fns, return RGB + count."""
        rgb = cv2.cvtColor(self.rectifier.apply(bgr), cv2.COLOR_BGR2RGB)
        n_total = 0
        for fn in self.detect_fns:
            rgb, k = fn(rgb)
            n_total += k
        return rgb, n_total


@dataclasses.dataclass
class RigPipeline:
    """rectify -> detect -> [pose_estimator] + [gripper_estimator].

    Both estimators consume the same ``Detections``, so the marker pass
    runs once per frame regardless of which estimators are wired in.
    Either estimator may be ``None`` to disable that branch — this
    degrades to a single-component pipeline without code duplication.

    Marker-set overlap: if the gripper anchor marker is also in the
    pose-estimator's world layout, a single shared detector is enough.
    Otherwise the detector should be configured with both marker sets
    so its ``Detections`` cover both consumers.
    """

    rectifier: Rectifier
    detector: Detector
    pose_estimator: PoseEstimator | None = None
    gripper_estimator: GripperEstimator | None = None

    def process(
        self, frame: np.ndarray,
    ) -> tuple[np.ndarray, Detections, FramePose | None, GripperState | None]:
        rectified = self.rectifier.apply(frame)
        detections = self.detector(rectified)
        K = self.rectifier.K_pinhole
        pose = (
            self.pose_estimator(detections, K)
            if self.pose_estimator is not None else None
        )
        gripper = (
            self.gripper_estimator(rectified, detections, K)
            if self.gripper_estimator is not None else None
        )
        return rectified, detections, pose, gripper
