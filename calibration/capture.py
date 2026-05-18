#!/usr/bin/env python3
"""Capture frames from the Insta360 driver with live marker-detection overlay.

Subscribes to a ROS2 image topic (default ``/dual_fisheye/image``) and an
optional CameraInfo topic, runs one or more marker detectors per frame,
and serves a viser preview at ``http://localhost:<port>``. Click the
"Capture" button to save the current frame. Intrinsics, when received,
are saved alongside as ``camera_intrinsics.npz``.

Target types match ``replay_video``: aruco | charuco | apriltag |
apriltag_grid | multi.

Source your ROS2 environment first, then run::

    # AprilGrid (default)
    python -m calibration.capture \\
        --marker-configs config/apriltag_board.yaml

    # Combine ArUco box markers + AprilGrid in one pass
    python -m calibration.capture --target-type multi \\
        --marker-configs config/aruco_set.yaml config/apriltag_board.yaml
"""

from __future__ import annotations

import dataclasses
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal

import cv2
import numpy as np
import rclpy
import tyro
import viser
import yaml
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

from core.markers import (
    create_charuco_board,
    detect_aruco_markers,
    detect_charuco_corners,
    draw_aruco_overlay,
    draw_charuco_overlay,
    load_apriltag_grid_configs,
    load_charuco_board_configs,
    load_marker_configs,
    resolve_aruco_dict,
)

TargetType = Literal["aruco", "charuco", "apriltag", "apriltag_grid", "multi"]

DetectFn = Callable[[np.ndarray], tuple[np.ndarray, int]]


def _default_output_dir() -> Path:
    return Path.cwd() / f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


@dataclasses.dataclass
class Args:
    """Insta360 marker-capture configuration."""

    image_topic: str = "/dual_fisheye/image"
    """ROS2 image topic to subscribe to."""

    camera_info_topic: str = "/camera_info"
    """ROS2 CameraInfo topic. Optional; intrinsics are saved on first message."""

    output_dir: Path = dataclasses.field(default_factory=_default_output_dir)
    """Directory to save captured images. Defaults to capture_<datetime>."""

    target_type: TargetType = "apriltag_grid"
    """aruco | charuco | apriltag | apriltag_grid | multi."""

    tag_dictionary: str = "DICT_APRILTAG_36h11"
    """OpenCV aruco dictionary for loose `apriltag` mode."""

    marker_configs: tuple[Path, ...] = ()
    """Marker preset YAML paths. Required for everything except `apriltag`."""

    port: int = 8085
    """Viser web viewer port."""


class CaptureNode(Node):
    def __init__(self, args: Args) -> None:
        super().__init__("capture_node")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = args.output_dir

        self.bridge = CvBridge()
        self._lock = threading.Lock()
        self._intrinsics_saved = False
        self._pending_capture = False
        self._count = 0
        self._camera_matrix: np.ndarray | None = None
        self._dist_coeffs: np.ndarray | None = None

        self._detect_fns: list[DetectFn] = self._build_detector_fns(args)
        self.get_logger().info(f"Detectors active: {len(self._detect_fns)}")

        self._server = viser.ViserServer(port=args.port)
        self._image_handle = self._server.gui.add_image(
            np.zeros((16, 16, 3), dtype=np.uint8), label="Camera"
        )
        self._count_display = self._server.gui.add_number(
            "Images captured", initial_value=0, disabled=True
        )
        capture_btn = self._server.gui.add_button("Capture", icon=viser.Icon.CAMERA)

        @capture_btn.on_click
        def _(_: object) -> None:
            with self._lock:
                self._pending_capture = True

        self.get_logger().info(f"Viser preview: http://localhost:{args.port}")
        self.get_logger().info(f"Saving to {self.output_dir}")

        self.create_subscription(
            Image, args.image_topic, self._image_cb, qos_profile_sensor_data
        )
        self.create_subscription(
            CameraInfo, args.camera_info_topic, self._info_cb, qos_profile_sensor_data
        )

    # -----------------------------------------------------------------------
    # Detector factories — mirror replay_video's, with charuco optionally
    # consuming live intrinsics from self.
    # -----------------------------------------------------------------------

    def _make_aruco_detector(self, config_path: Path) -> DetectFn:
        markers = load_marker_configs(config_path)
        if not markers:
            raise RuntimeError(f"No markers defined in {config_path}")
        by_dict: dict[str, set[int]] = {}
        for mc in markers.values():
            by_dict.setdefault(mc.dictionary, set()).add(mc.id)
        if len(by_dict) > 1:
            raise RuntimeError(
                f"{config_path}: markers use multiple dictionaries; split per file."
            )
        dict_name, allowed_ids = next(iter(by_dict.items()))
        dict_id = resolve_aruco_dict(dict_name)
        self.get_logger().info(
            f"ArUco ({config_path.name}, {dict_name}, IDs {sorted(allowed_ids)})"
        )

        def detect(rgb: np.ndarray) -> tuple[np.ndarray, int]:
            corners, ids = detect_aruco_markers(
                rgb, marker_dict=dict_id, allowed_ids=allowed_ids
            )
            if corners is not None:
                return draw_aruco_overlay(rgb, corners, ids), int(len(ids))
            return rgb, 0

        return detect

    def _make_charuco_detector(self, config_path: Path) -> DetectFn:
        boards = load_charuco_board_configs(config_path)
        if not boards:
            raise RuntimeError(f"No ChArUco boards defined in {config_path}")
        name, board_cfg = next(iter(boards.items()))
        board = create_charuco_board(board_cfg)
        self.get_logger().info(
            f"ChArUco ({config_path.name}, '{name}', "
            f"{board_cfg.squares_x}x{board_cfg.squares_y} squares)"
        )

        def detect(rgb: np.ndarray) -> tuple[np.ndarray, int]:
            ch_corners, ch_ids, mkr_corners, mkr_ids = detect_charuco_corners(
                rgb, board, self._camera_matrix, self._dist_coeffs
            )
            if ch_corners is not None:
                return (
                    draw_charuco_overlay(rgb, ch_corners, ch_ids, mkr_corners, mkr_ids),
                    int(len(ch_ids)),
                )
            return rgb, 0

        return detect

    def _make_apriltag_grid_detector(self, config_path: Path) -> DetectFn:
        grids = load_apriltag_grid_configs(config_path)
        if not grids:
            raise RuntimeError(f"No apriltag_grid defined in {config_path}")
        name, grid_cfg = next(iter(grids.items()))
        allowed_ids = set(grid_cfg.tag_ids)
        dict_id = grid_cfg.cv2_dictionary
        last_id = grid_cfg.start_id + len(allowed_ids) - 1
        self.get_logger().info(
            f"AprilGrid ({config_path.name}, '{name}', "
            f"{grid_cfg.tag_cols}x{grid_cfg.tag_rows} {grid_cfg.dictionary}, "
            f"IDs {grid_cfg.start_id}..{last_id})"
        )

        def detect(rgb: np.ndarray) -> tuple[np.ndarray, int]:
            corners, ids = detect_aruco_markers(
                rgb, marker_dict=dict_id, allowed_ids=allowed_ids
            )
            if corners is not None:
                return draw_aruco_overlay(rgb, corners, ids), int(len(ids))
            return rgb, 0

        return detect

    def _make_apriltag_dict_detector(self, dict_name: str) -> DetectFn:
        dict_id = resolve_aruco_dict(dict_name)
        self.get_logger().info(f"AprilTag (loose, {dict_name})")

        def detect(rgb: np.ndarray) -> tuple[np.ndarray, int]:
            corners, ids = detect_aruco_markers(rgb, marker_dict=dict_id)
            if corners is not None:
                return draw_aruco_overlay(rgb, corners, ids), int(len(ids))
            return rgb, 0

        return detect

    def _detectors_from_config(self, path: Path) -> list[DetectFn]:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        fns: list[DetectFn] = []
        if raw.get("markers"):
            fns.append(self._make_aruco_detector(path))
        if raw.get("apriltag_grid"):
            fns.append(self._make_apriltag_grid_detector(path))
        if raw.get("charuco"):
            fns.append(self._make_charuco_detector(path))
        if not fns:
            raise RuntimeError(
                f"{path}: no recognised sections (markers, apriltag_grid, charuco)"
            )
        return fns

    def _build_detector_fns(self, args: Args) -> list[DetectFn]:
        if args.target_type == "apriltag":
            return [self._make_apriltag_dict_detector(args.tag_dictionary)]

        if args.target_type == "multi":
            if not args.marker_configs:
                raise RuntimeError(
                    "--marker-configs is required for multi mode (pass one or more YAMLs)"
                )
            fns: list[DetectFn] = []
            for path in args.marker_configs:
                fns.extend(self._detectors_from_config(path))
            return fns

        if not args.marker_configs:
            raise RuntimeError(
                f"--marker-configs is required for target-type={args.target_type}"
            )
        if len(args.marker_configs) > 1:
            raise RuntimeError(
                f"target-type={args.target_type} expects exactly one --marker-configs; "
                "use target-type=multi to combine."
            )
        path = args.marker_configs[0]
        if args.target_type == "aruco":
            return [self._make_aruco_detector(path)]
        if args.target_type == "charuco":
            return [self._make_charuco_detector(path)]
        if args.target_type == "apriltag_grid":
            return [self._make_apriltag_grid_detector(path)]
        raise ValueError(f"unknown target_type: {args.target_type}")

    # -----------------------------------------------------------------------
    # ROS callbacks
    # -----------------------------------------------------------------------

    def _info_cb(self, msg: CameraInfo) -> None:
        K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        dist = np.array(msg.d, dtype=np.float64)
        self._camera_matrix = K
        self._dist_coeffs = dist
        if not self._intrinsics_saved:
            path = self.output_dir / "camera_intrinsics.npz"
            np.savez(
                str(path),
                camera_matrix=K,
                dist_coeffs=dist,
                width=msg.width,
                height=msg.height,
            )
            self._intrinsics_saved = True
            self.get_logger().info(
                f"Saved intrinsics to {path}  "
                f"(fx={K[0, 0]:.1f} fy={K[1, 1]:.1f} cx={K[0, 2]:.1f} cy={K[1, 2]:.1f})"
            )

    def _image_cb(self, msg: Image) -> None:
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        raw_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        annotated = raw_rgb
        for fn in self._detect_fns:
            annotated, _ = fn(annotated)
        self._image_handle.image = annotated

        with self._lock:
            should_capture = self._pending_capture
            self._pending_capture = False

        if should_capture:
            self._save_frame(raw_rgb)

    def _save_frame(self, rgb: np.ndarray) -> None:
        filepath = self.output_dir / f"capture_{self._count:04d}.png"
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(filepath), bgr)
        self._count += 1
        self._count_display.value = self._count
        self.get_logger().info(f"[{self._count}] Saved {filepath}")


def main(args: Args) -> None:
    rclpy.init()
    node = CaptureNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main(tyro.cli(Args))
