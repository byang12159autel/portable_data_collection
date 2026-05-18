"""ArUco / ChArUco / AprilTag detection and YAML config loading.

All detectors run through OpenCV's ``cv2.aruco`` module. AprilTag families
(``DICT_APRILTAG_36h11`` etc.) are predefined ArUco dictionaries in OpenCV,
so the same ``detect_aruco_markers`` / ``draw_aruco_overlay`` helpers work
for them. This is what we use for calib.io AprilGrid targets (which
``pupil_apriltags`` failed to detect due to polarity / border-rendering
differences from the standard apriltag3 format).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import cv2
import numpy as np
import yaml

ARUCO_DICT_MAP: dict[str, int] = {
    name: getattr(cv2.aruco, name) for name in dir(cv2.aruco) if name.startswith("DICT_")
}


def resolve_aruco_dict(name: str) -> int:
    """Look up the OpenCV enum value for an ArUco / AprilTag dictionary name."""
    if name not in ARUCO_DICT_MAP:
        raise ValueError(
            f"Unknown ArUco dictionary '{name}'. Valid options: {sorted(ARUCO_DICT_MAP)}"
        )
    return ARUCO_DICT_MAP[name]


@dataclasses.dataclass(frozen=True)
class MarkerConfig:
    id: int
    size: float
    dictionary: str
    label: str = ""

    @property
    def cv2_dictionary(self) -> int:
        if self.dictionary not in ARUCO_DICT_MAP:
            raise ValueError(
                f"Unknown ArUco dictionary '{self.dictionary}'. "
                f"Valid options: {sorted(ARUCO_DICT_MAP)}"
            )
        return ARUCO_DICT_MAP[self.dictionary]


@dataclasses.dataclass(frozen=True)
class CharucoBoardConfig:
    squares_x: int
    squares_y: int
    square_length: float
    marker_length: float
    dictionary: str
    label: str = ""

    @property
    def cv2_dictionary(self) -> int:
        if self.dictionary not in ARUCO_DICT_MAP:
            raise ValueError(
                f"Unknown ArUco dictionary '{self.dictionary}'. "
                f"Valid options: {sorted(ARUCO_DICT_MAP)}"
            )
        return ARUCO_DICT_MAP[self.dictionary]


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_marker_configs(path: Path) -> dict[str, MarkerConfig]:
    """Load named ArUco marker presets from the ``markers:`` section of a YAML file.

    Unknown fields per marker (e.g. ``T_box_marker``) are silently ignored so
    YAMLs that carry extra metadata for downstream pose math still load.
    """
    raw = _load_yaml(path)
    markers = raw.get("markers", {})
    known = {f.name for f in dataclasses.fields(MarkerConfig)}
    return {
        name: MarkerConfig(**{k: v for k, v in fields.items() if k in known})
        for name, fields in markers.items()
    }


def load_charuco_board_configs(path: Path) -> dict[str, CharucoBoardConfig]:
    """Load named ChArUco board presets from the ``charuco:`` section of a YAML file."""
    raw = _load_yaml(path)
    boards = raw.get("charuco", {})
    return {name: CharucoBoardConfig(**fields) for name, fields in boards.items()}


def load_named_marker(
    path: Path, name: str | None = None,
) -> tuple[MarkerConfig, int]:
    """Load one marker by name from a YAML's ``markers:`` section.

    With ``name=None`` returns the first entry (handy when the YAML only
    has one). Returns ``(MarkerConfig, cv2_dict_id)``; raises
    ``SystemExit`` if the YAML has no markers or the named entry is
    missing.
    """
    markers = load_marker_configs(path)
    if not markers:
        raise SystemExit(f"no markers defined in {path}")
    if name is None:
        name, cfg = next(iter(markers.items()))
    else:
        if name not in markers:
            raise SystemExit(
                f"marker '{name}' not in {path}; available: {list(markers)}"
            )
        cfg = markers[name]
    return cfg, cfg.cv2_dictionary


def detect_aruco_markers(
    img: np.ndarray,
    marker_dict: int = cv2.aruco.DICT_4X4_50,
    allowed_ids: set[int] | None = None,
) -> tuple[tuple | None, np.ndarray | None]:
    """Detect ArUco markers in an RGB image.

    Returns ``(corners, ids)`` where ``corners`` is a tuple of ``(1, 4, 2)`` float32
    arrays and ``ids`` is an ``(N, 1)`` int32 array; both ``None`` if no markers
    pass the filter.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(marker_dict)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, _ = detector.detectMarkers(img)
    if ids is None:
        return None, None

    if allowed_ids is not None:
        mask = np.isin(ids.flatten(), list(allowed_ids))
        if not np.any(mask):
            return None, None
        corners = tuple(c for c, m in zip(corners, mask) if m)
        ids = ids[mask].reshape(-1, 1)

    return corners, ids


def marker_object_points(marker_size: float) -> np.ndarray:
    """3D corners of a planar square marker centered at the origin.

    Order: top-left, top-right, bottom-right, bottom-left (OpenCV convention).
    """
    half = marker_size / 2.0
    return np.array(
        [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]],
        dtype=np.float32,
    )


def create_charuco_board(config: CharucoBoardConfig) -> cv2.aruco.CharucoBoard:
    aruco_dict = cv2.aruco.getPredefinedDictionary(config.cv2_dictionary)
    return cv2.aruco.CharucoBoard(
        (config.squares_x, config.squares_y),
        config.square_length,
        config.marker_length,
        aruco_dict,
    )


def detect_charuco_corners(
    img: np.ndarray,
    board: cv2.aruco.CharucoBoard,
    camera_matrix: np.ndarray | None = None,
    dist_coeffs: np.ndarray | None = None,
    min_corners: int = 4,
) -> tuple[np.ndarray | None, np.ndarray | None, tuple | None, np.ndarray | None]:
    """Detect ChArUco corners in an RGB image."""
    detector_params = cv2.aruco.DetectorParameters()
    charuco_params = cv2.aruco.CharucoParameters()
    if camera_matrix is not None:
        charuco_params.cameraMatrix = camera_matrix.astype(np.float64)
    if dist_coeffs is not None:
        charuco_params.distCoeffs = dist_coeffs.astype(np.float64)

    detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params)
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(img)

    if charuco_ids is None or len(charuco_ids) < min_corners:
        return None, None, None, None

    return charuco_corners, charuco_ids, marker_corners, marker_ids


def charuco_object_points(
    board: cv2.aruco.CharucoBoard,
    charuco_ids: np.ndarray,
) -> np.ndarray:
    """3D positions of detected ChArUco corners, centered at the board face center."""
    all_corners_3d = board.getChessboardCorners()
    sq = board.getSquareLength()
    n_cols, n_rows = board.getChessboardSize()
    center = np.array([n_cols * sq / 2.0, n_rows * sq / 2.0, 0.0], dtype=np.float64)
    ids_flat = charuco_ids.flatten()
    return (all_corners_3d[ids_flat] - center).astype(np.float32)


def draw_aruco_overlay(
    img: np.ndarray,
    corners: tuple,
    ids: np.ndarray,
) -> np.ndarray:
    """Draw detected ArUco markers on a copy of ``img``."""
    out = cv2.aruco.drawDetectedMarkers(img.copy(), corners, ids, borderColor=(0, 255, 0))
    for c in corners:
        pts = c[0].astype(np.int32)
        cv2.polylines(out, [pts], isClosed=True, color=(0, 255, 0), thickness=4)
    return out


def draw_charuco_overlay(
    img: np.ndarray,
    charuco_corners: np.ndarray,
    charuco_ids: np.ndarray,
    marker_corners: tuple | None = None,
    marker_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Draw detected ChArUco corners (and optionally ArUco markers) on a copy of ``img``."""
    out = img.copy()
    if marker_corners is not None and marker_ids is not None:
        cv2.aruco.drawDetectedMarkers(out, marker_corners, marker_ids, borderColor=(255, 165, 0))
    cv2.aruco.drawDetectedCornersCharuco(out, charuco_corners, charuco_ids, cornerColor=(0, 255, 0))
    return out


# ---------------------------------------------------------------------------
# AprilGrid (calib.io / Kalibr-style)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class AprilGridConfig:
    """calib.io / Kalibr-style AprilGrid: a rigid grid of AprilTags."""

    tag_cols: int
    """Number of tags in the x dimension (calib.io 'Columns')."""

    tag_rows: int
    """Number of tags in the y dimension (calib.io 'Rows')."""

    tag_size: float
    """Side length of one tag in metres (calib.io 'Tag Size')."""

    tag_spacing: float
    """Gap between adjacent tags as a fraction of ``tag_size`` (Kalibr ratio).
    Example: 15 mm gap on a 50 mm tag = 0.30."""

    dictionary: str = "DICT_APRILTAG_36h11"
    """OpenCV aruco dictionary name. calib.io's default for AprilGrid is 36h11."""

    start_id: int = 0
    """First tag ID used by the grid (calib.io 'Start Id'). The grid occupies
    IDs ``[start_id, start_id + tag_cols * tag_rows)``."""

    label: str = ""

    @property
    def cv2_dictionary(self) -> int:
        return resolve_aruco_dict(self.dictionary)

    @property
    def tag_ids(self) -> range:
        """All tag IDs belonging to this grid."""
        return range(self.start_id, self.start_id + self.tag_cols * self.tag_rows)

    @property
    def stride(self) -> float:
        """Centre-to-centre distance between adjacent tags (m)."""
        return self.tag_size * (1.0 + self.tag_spacing)


def load_apriltag_grid_configs(path: Path) -> dict[str, AprilGridConfig]:
    """Load named AprilGrid presets from the ``apriltag_grid:`` section of a YAML file."""
    raw = _load_yaml(path)
    grids = raw.get("apriltag_grid", {})
    return {name: AprilGridConfig(**fields) for name, fields in grids.items()}


def apriltag_grid_object_points(config: AprilGridConfig) -> dict[int, np.ndarray]:
    """3D positions of each tag's 4 corners in the board frame (Kalibr convention).

    Verified against Kalibr's ``GridCalibrationTargetAprilgrid::createGridPoints``
    (``aslam_cv/aslam_cameras_april/src/GridCalibrationTargetAprilgrid.cpp``):

      - **Origin** at the bottom-left corner of tag ``start_id`` (the
        (col=0, row=0) tag).
      - **x axis** along columns (rightward).
      - **y axis** along rows pointing **up** (so ``row=0`` is the bottom
        row of the board).
      - z = 0 (board lies in the xy plane).

    Tag IDs increment row-major from the bottom-left tag::

        tag_id = start_id + row * tag_cols + col   # row 0 is the bottom row

    Per-tag corner order is OpenCV ArUco's ``[TL, TR, BR, BL]`` (matches
    ``cv2.aruco.ArucoDetector.detectMarkers`` output), which is the reverse
    of Kalibr's internal ``[BL, BR, TR, TL]`` pIdx storage — same four
    points, swapped order, so ``solvePnP(obj_pts, image_corners)`` works
    directly without reordering the detected corners.

    Returns ``{tag_id: (4, 3) float32 ndarray}`` for every tag in the grid.
    """
    stride = config.stride
    s = config.tag_size
    points: dict[int, np.ndarray] = {}
    for r in range(config.tag_rows):
        for c in range(config.tag_cols):
            tag_id = config.start_id + r * config.tag_cols + c
            x0 = c * stride       # left edge of this tag
            y0 = r * stride       # bottom edge of this tag (y-up)
            points[tag_id] = np.array(
                [
                    [x0,     y0 + s, 0.0],  # TL — top edge has the higher y
                    [x0 + s, y0 + s, 0.0],  # TR
                    [x0 + s, y0,     0.0],  # BR
                    [x0,     y0,     0.0],  # BL
                ],
                dtype=np.float32,
            )
    return points


