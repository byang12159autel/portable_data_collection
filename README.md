# portable_data_collection

Marker-based pose estimation for an Insta360 camera, with offline video
replay for verifying detection.

## Layout

The pipeline is split into stage subpackages so each step can be swapped
independently. Stage protocols live in `pose_calibration/pipeline.py`;
each subpackage holds the implementations that satisfy them.

```
pose_calibration/
  pipeline.py          Stage Protocols (FrameSource, Calibrator, Rectifier,
                       Detector, PoseEstimator) + Pipeline composition class
  markers/             Marker detection — implements Detector
    detect.py          ArUco / ChArUco / AprilTag / AprilGrid + YAML loaders
  calibration/         Camera calibration — implements Calibrator + Rectifier
    fisheye.py         Per-lens cv2.fisheye.calibrate (primary path)
    two_stage.py       Equidistant unwrap + pinhole refine (fragile-fisheye fallback)
    pinhole.py         Standard cv2.calibrateCamera for already-unwarped inputs
    auto.py            Sweep thresholds, pick first acceptable result
    rectify.py         Fisheye → virtual pinhole LUT; loads any of the above
  camera/              General camera I/O
    convert.py         .insv → two per-lens .mp4s (X4/X5 store one stream per lens)
    inspect.py         ffprobe summary + per-stream frame0 dump
    split.py           Side-by-side splitter for the live USB stream (ROS path)
  pose/                Camera-pose estimation — implements PoseEstimator
    known_board.py     AprilGrid with layout from config; pooled PnP
    learned_layout.py  Independent ArUco markers; layout learned from co-visibility
  apps/                Entry-point scripts that compose a Pipeline
    replay_video.py    Replay any video with detection overlays in viser
    replay_insta.py    End-to-end Insta360: rectify + detect, side-by-side viser
    capture_node.py    ROS2 capture node + viser preview + capture button
scripts/
  debug/               One-off investigation scripts
  bench/               Sub-pixel detection benchmark
config/                Marker / board YAMLs
data/                  Sample videos
ros2_ws/src/
  insta360_ros_driver/ Camera driver (ai4ce/insta360_ros_driver)
pixi.toml              Conda + pip env spec (robostack-humble + viser + tyro)
```

To switch a stage's implementation, change one import in the calling
`apps/` script (or write a new class against the relevant protocol and
drop it into the matching subpackage).

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
pixi run python -m pose_calibration.apps.replay_video \
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

## Offline pipeline from `.insv` recordings

Record on the camera (no ROS, no USB tether), then process the file.
Data flows through six stages, with calibration done once per physical
camera and reused for every subsequent recording.

```
1. raw .insv                    (camera SD card; two H.265 streams + IMU)
        |
        | ffmpeg -map 0:v:0 / 0:v:1   (convert.py)
        v
2. <stem>_lens0.mp4 + _lens1.mp4         (two per-lens fisheye videos)
        |
        | board recording + cv2.fisheye.calibrate   (calibrate.py)
        v                                            [ONE-TIME, per camera]
3. insta360_intrinsics.npz       (K, D per lens; fisheye model)
        |
        | cv2.fisheye.initUndistortRectifyMap + cv2.remap   (rectify.py)
        v
4. front_rectified.mp4 + back_rectified.mp4 + pinhole_intrinsics.npz
        |                                            (virtual pinhole at chosen FOV)
        | cv2.aruco / cv2.aruco.CharucoDetector       (detect_marker.py)
        v
5. per-frame marker corners (pixels) + IDs
        |
        | cv2.solvePnP with pinhole K, distCoeffs=0            (future: compute_pose.py)
        v
6. T_camera_marker per tag per frame   (and T_world_camera if markers are fixed)
```

### Stages in detail

**1. Raw `.insv`** — the camera writes an MP4-like container with two
independent H.265 video streams (one per lens; X4/X5 are 1920x1920 @
30 fps each) plus an audio track and an Insta360-proprietary IMU box.
Verify the dual-stream layout with `ffprobe -show_entries stream=...`.

**2. Demux to per-lens mp4s** —
`pose_calibration.camera.convert` runs ffmpeg with `-map 0:v:0` and
`-map 0:v:1` as a stream-copy (no re-encode). After this step each
lens behaves like an ordinary fisheye video; the 360°/dual-fisheye
nature of the source is gone.

```bash
pixi run python -m pose_calibration.camera.convert \
    --input data/VID_20260515_..._00_001.insv
```

Open `_lens0.mp4` and `_lens1.mp4` in VLC to figure out which is the
screen-side ("front") lens on your unit before feeding them downstream.

**3. Per-lens fisheye calibration** (one-time) — record one `.insv` of
a planar AprilGrid / ChArUco board moved across each lens's FOV. For
each per-lens video, `calibrate.py` walks the frames, detects board
corners, builds `(obj_pts, img_pts)` correspondences, and calls
`cv2.fisheye.calibrate` to fit:

- `K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]` (focal length +
  principal point)
- `D = [k1, k2, k3, k4]` (fisheye distortion, OpenCV's equidistant +
  polynomial model)

Output is a single `insta360_intrinsics.npz` carrying `K_front`,
`D_front`, `K_back`, `D_back`, `image_size`, plus the per-lens RMS
reprojection error. Re-running with only `--front-video` or only
`--back-video` preserves the other lens's entries in the npz — useful
for splitting calibration across multiple recordings.

```bash
pixi run python -m pose_calibration.calibration.fisheye \
    --front-video data/calib_lens0.mp4 \
    --back-video  data/calib_lens1.mp4 \
    --marker-config config/apriltag_board.yaml \
    --output data/insta360_intrinsics.npz
```

A good calibration recording covers the full FOV of each lens with
diverse board positions, tilts, and distances — see the comments in
`calibrate.py` and the **Future** section below for the FOV-coverage
limitation we inherit by going single-pinhole.

#### Auto-sweep when thresholds aren't obvious

When you don't know good values for `--frame-stride`,
`--min-corners-per-view`, and `--min-spread-frac`, or when a `calibrate.py`
run failed with an obscure `cv2` assertion, use `auto_calibrate.py`. It
demuxes the `.insv` (skipped if the per-lens mp4s already exist), runs
one expensive permissive collection pass per lens, then sweeps a grid
of progressively-loosened threshold combinations, stopping at the first
result that meets quality criteria (RMS < 2 px, focal in [400, 800],
`|fy/fx − 1| < 0.15`, `|k1| < 1`). If nothing meets them, it still
saves the best result by RMS so you can decide.

```bash
pixi run python -m pose_calibration.calibration.auto \
    --insv data/<recording>.insv \
    --marker-config config/apriltag_board.yaml \
    --output data/insta360_intrinsics.npz
```

Override the acceptance thresholds via `--target-rms`, `--focal-min`,
`--focal-max`, `--fy-fx-tolerance`, `--k1-max`. Override the sweep
grid by editing `SWEEP_GRID` at the top of `auto_calibrate.py`.

#### Two-stage fallback when fisheye calibration won't converge

`cv2.fisheye.calibrate` is fragile when the recording doesn't get
corners to the **periphery** of the fisheye circle — the high-order
fisheye distortion coefficients stay unconstrained and the optimizer
slides into garbage (telltale signs: RMS in the hundreds of pixels,
fx far from the equidistant prediction `W/π`, distortion coefficients
oscillating with large magnitude). The corner-coverage plot
(see `debug_corner_coverage.py`) tells you whether this is your
problem: green dots clustered in the centre means yes.

`two_stage_calibrate.py` works around this by sidestepping fisheye-
model fitting entirely:

1. **Stage 1** rectifies each frame to a virtual pinhole using the
   equidistant fisheye assumption (`f = W/π`, no parameters to fit).
2. **Stage 2** runs the standard `cv2.calibrateCamera` (pinhole model
   with 5 distortion coefficients) on the board corners detected in
   the rough-pinhole frames. The pinhole solver tolerates limited
   coverage far better than the fisheye one.

The combined model captures the actual lens behavior even though
neither stage alone does. Downstream `rectify.py` and `replay_insta.py`
auto-detect two-stage results in the `.npz` and apply the composed
LUT — no other code paths change.

```bash
pixi run python -m pose_calibration.calibration.two_stage \
    --front-video data/<recording>_lens0.mp4 \
    --back-video  data/<recording>_lens1.mp4 \
    --marker-config config/apriltag_board.yaml \
    --output data/insta360_intrinsics.npz \
    --fov-deg 110 --pinhole-width 1280 --pinhole-height 1280
```

Caveat: the chosen `--fov-deg` and `--pinhole-{width,height}` are
**baked in** at calibration time. Rectifying at a different FOV/size
later requires re-running this script. (Inspired by the Gaussian-
Splatting community's Insta360 X5 workflow, which lets COLMAP /
Metashape refine pinhole intrinsics after fisheye unwrap.)

**4. Fisheye → virtual pinhole rectification** — for each frame of a
scene recording, `rectify.py` applies a precomputed
`cv2.fisheye.initUndistortRectifyMap` LUT that re-projects the fisheye
view onto a synthetic pinhole. The rectified frame behaves like a
standard pinhole camera with intrinsics `K_pinhole` (saved alongside
as `pinhole_intrinsics.npz`) and **zero distortion**. Straight lines
in the world are now straight in the image — necessary for AprilTag
corner refinement and OpenCV's pinhole-model `solvePnP`.

Cost: the chosen FOV (default 110°) clips ~half of each lens's ~190°
FOV; markers off-axis fall outside the rectified frame.

```bash
pixi run python -m pose_calibration.calibration.rectify \
    --front-video data/scene_lens0.mp4 \
    --back-video  data/scene_lens1.mp4 \
    --intrinsics data/insta360_intrinsics.npz \
    --output-dir data/rectified \
    --fov-deg 110
```

**5. Marker detection on rectified frames** — same detectors as
`replay_video.py` (`cv2.aruco.ArucoDetector`,
`cv2.aruco.CharucoDetector`). Returns `corners` (pixel-space quads per
tag) + `ids` (tag IDs from the dictionary).

For a quick interactive preview that runs steps 4 + 5 together with no
intermediate mp4s, use `replay_insta.py`:

```bash
pixi run python -m pose_calibration.apps.replay_insta \
    --front-video data/scene_lens0.mp4 \
    --back-video  data/scene_lens1.mp4 \
    --intrinsics data/insta360_intrinsics.npz \
    --target-type multi \
    --marker-configs config/aruco_set.yaml config/apriltag_board.yaml
```

Viser preview at `http://localhost:8085` shows the rectified front and
back side-by-side with detection overlays and a live "Detected" count.

**6. Pose estimation** *(not yet implemented in this repo — pending)* —
once corners + `K_pinhole` are in hand, the math is one `solvePnP` call
per tag:

```python
obj_pts = np.array([[-s/2,  s/2, 0],
                    [ s/2,  s/2, 0],
                    [ s/2, -s/2, 0],
                    [-s/2, -s/2, 0]], dtype=np.float32)  # s = marker size
ok, rvec, tvec = cv2.solvePnP(
    obj_pts,                   # marker frame
    detected_corners,          # pixels in the rectified frame
    K_pinhole,                 # from pinhole_intrinsics.npz
    distCoeffs=np.zeros(5),    # rectified frame is undistorted
    flags=cv2.SOLVEPNP_IPPE_SQUARE,
)
# (rvec, tvec) -> T_camera_marker
```

For a board (AprilGrid / ChArUco), stack all detected tag corners with
their known 3D positions on the board and one `solvePnP` gives
`T_camera_board`. If markers are at known fixed world positions,
invert to get `T_world_camera`.

### Sanity check at each stage

| Stage | What to verify |
|---|---|
| 2 demux           | Open `_lens0.mp4` in VLC — see a fisheye circle. |
| 3 calibrate       | RMS < ~1 px; fx ≈ 580–650 for a 190° FOV lens. |
| 4 rectify         | Straight-edged objects in the scene are straight in the rectified frame. |
| 5 detect          | `replay_insta`'s "Detected" count goes up when a marker enters the rectified FOV. |
| 6 PnP             | Re-project marker corners with the recovered `(rvec, tvec)` — should land within ~1 px of detection. |

### Future: multi-virtual-pinhole per lens

A single rectified pinhole covers ~100° of each lens's ~190° FOV, so
markers off-axis fall outside the rectified frame and are missed — the
camera's 360° advantage is wasted. The planned upgrade is to project
each lens to **several** virtual pinholes per frame (e.g. four cube
faces) and run detection across all of them, then merge per-tag
detections back into a single body-frame pose. The single-pinhole
`Rectifier` class is the building block; a future `MultiRectifier`
will hold a list of them with non-identity `R` rotations.

## Live capture (ROS2)

After the insta360 driver is running and publishing on `/dual_fisheye/image`:

```bash
pixi run python -m pose_calibration.apps.capture_node \
    --target-type multi \
    --marker-configs config/aruco_set.yaml config/apriltag_board.yaml
```

Same viser preview; click **Capture** in the browser to save the current
raw (un-annotated) frame as PNG. If CameraInfo is available, intrinsics
are saved alongside as `camera_intrinsics.npz` on first message.
