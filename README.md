# portable_data_collection

Marker-based pose estimation for an Insta360 camera, with offline video
replay for verifying detection.

## Layout

```
pose_calibration/      Python package (marker detection, capture node, replay)
  detect_marker.py     ArUco / ChArUco / AprilTag / AprilGrid detection + YAML loaders
  capture_node.py      ROS2 node: subscribes to insta360 image topic, viser preview, capture button
  replay_video.py      Replay a recorded video with detection overlays in viser
  compute_pose.py      Eye-to-hand calibration (legacy avantbot imports; pending rewrite)
config/                Marker / board YAMLs
data/                  Sample videos
ros2_ws/src/
  insta360_ros_driver/ Camera driver (ai4ce/insta360_ros_driver)
pixi.toml              Conda + pip env spec (robostack-humble + viser + tyro)
```

## Setup

```bash
# Clone with the insta360 driver submodule
git clone --recurse-submodules <repo-url>
# (or, if already cloned: git submodule update --init --recursive)

pixi install                                  # one-time conda/pip env
pixi shell
# optional: build the insta360 C++ driver inside ros2_ws (needs Insta360 SDK)
pixi run build
```

`ros-humble-imu-tools` is not in `pixi.toml` (version conflict on robostack);
install it via apt if you need the bringup launch's IMU filter.

## Replay a recorded video with detection overlays

Verify both ArUco markers (on a calibration box) **and** an AprilGrid board
in the same frame:

```bash
pixi run python -m pose_calibration.replay_video \
    --video data/iphone_charuco+april.mov --target-type multi \
    --marker-configs config/aruco_set.yaml config/apriltag_board.yaml
```

Opens a viser preview at `http://localhost:8085` with Play/Pause + frame
slider. Drag the slider to seek; the "Detected" readout shows total marker
count across all active detectors.

## Other target types

| `--target-type`  | Config schema                         | Notes                              |
|------------------|---------------------------------------|------------------------------------|
| `apriltag_grid`  | `apriltag_grid:` (Kalibr-style grid)  | Default. One config.               |
| `aruco`          | `markers:` (list of `{id, size, dict}`) | One config; one dictionary per file. |
| `charuco`        | `charuco:` (board params)             | One config.                        |
| `apriltag`       | none — uses `--tag-dictionary`        | Loose, any AprilTag in the dict.   |
| `multi`          | any of the above, possibly several    | Auto-dispatches per section.       |

## Live capture (ROS2)

After the insta360 driver is running and publishing on `/dual_fisheye/image`:

```bash
pixi run python -m pose_calibration.capture_node \
    --target-type multi \
    --marker-configs config/aruco_set.yaml config/apriltag_board.yaml
```

Same viser preview; click **Capture** in the browser to save the current
raw (un-annotated) frame as PNG. If CameraInfo is available, intrinsics
are saved alongside as `camera_intrinsics.npz` on first message.
