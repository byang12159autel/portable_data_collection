# Sub-pixel refinement queue — **COMPLETE (stop condition met)**

🏆 **Best at canonical 0.60 spacing: `upscale8x_norefine` + outlier-filter 20%, RMS = 0.960 px**
(view count 197 after dropping 49 worst-view outliers from 246).

The earlier 0.546 px result was at the empirically-fit 0.54 spacing; K matched
the 0.60-spacing K within 0.1%, so the canonical calibration uses 0.60
(user-confirmed physical 30mm gap on 50mm tag) and the 0.96 number.

Best calibration saved at
`data/insta360_calibration/lens0_combined_subpixel_best.npz`.

## Cache & speedup
- `bench_subpixel.py --build-cache` writes
  `lens0_combined_unwrap_cache_<size>_<fov>_<stride>.npy` (5.1 GB for the
  default config). Auto-detected by subsequent `--method` runs.
- Speedup: baseline_aruco_subpix went **4:10 → 0:20** (~12x). Higher
  upscale methods are detection-bound (the upscale itself dominates), so
  the cache gains less there (upscale8x is still ~6 min/run).
- `pose_history.py` shares the cache for fast pose precompute.

## Outlier filter results (upscale8x_norefine + drop top X% per-view RMS)
| filter | RMS | views dropped | K change vs filter=0 |
|---|---|---|---|
| 0% | 1.021 | 0 | — |
| 5% | 1.003 | 13 | tiny |
| 10% | 0.985 | 25 | small |
| 20% | 0.960 | 49 | small |

Diminishing returns; 20% is a reasonable stopping point.

## Ideas tried but failed
- `upscale16x_norefine` — detector breaks at 20480×20480 (0 views returned
  in 16 min). Hard upper limit on this approach.

## Open: marker-pose Z offset

Observation (visible in the pose viewer, 2026-05-16): per-frame
single-tag PnP places each marker ~30-60 mm *above* the board's z=0 plane,
i.e. between the board and the camera. Offset is roughly constant across
markers, with mild variation by board position.

Sample (frame 0, 5 tags):
| tag | computed (x, y, z) | GT (x, y, z) | z offset (mm) |
|---|---|---|---|
| 60 | (0.039, 0.506, 0.029) | (0.025, 0.505, 0) | 29 |
| 61 | (0.122, 0.509, 0.047) | (0.105, 0.505, 0) | 47 |
| 64 | (0.350, 0.509, 0.058) | (0.345, 0.505, 0) | 58 |
| 68 | (0.653, 0.508, 0.047) | (0.665, 0.505, 0) | 47 |
| 53 | (0.271, 0.431, 0.035) | (0.265, 0.425, 0) | 35 |

### Likely causes, ranked

1. **Focal-length / depth degeneracy on a planar target.** Calibration
   landed at `fx = 375.13` for a 1280×1280 unwrap at 110° FOV; the
   textbook focal is `(W/2)/tan(FOV/2) ≈ 448`. The ~16 % underestimate is
   the classic planar-target ambiguity (no depth variation in obj_pts
   means focal and per-view depth are weakly separated). With `f` low,
   every PnP estimates Z slightly smaller than reality → markers appear
   between camera (z≈0.9) and board (z=0).
2. **Tag-size mismatch.** If physical tags are slightly larger than the
   YAML's 50 mm (e.g. 51 mm), per-tag PnP would place each marker
   ~18 mm closer to camera — consistent with the smaller end of the
   observed offsets.
3. **Board flatness.** Center-of-board tags showing larger z than edges
   would indicate a mild bow; tag 64 (most central in this sample) does
   have the largest offset, so a small contribution from this is
   plausible.

### Things to try

- [ ] **Sweep `tag_size` analogously to `tag_spacing`** (see
  `debug_global_spacing.py`). Find the size that minimises the
  z-component of `T_board_marker − GT_marker` across many frames.
- [ ] **Re-shoot calibration with more depth variation.** The current
  videos keep the camera at ~0.5-0.9 m. Adding sweeps at 0.3 m and 1.5 m
  would break the focal-depth ambiguity.
- [ ] **Bundle-adjust tag positions and intrinsics jointly** (Kalibr-style).
  Both `tag_size`/`tag_spacing` and `K, D` get refined together against
  per-corner reprojection. Most principled fix; biggest implementation
  cost.
- [ ] **Check focal against a known-distance shot.** Take one frame with
  the board at a precisely measured distance and verify `fx` from the
  pixel size of a tag.

## tl;dr (read this first in the morning)

The overnight loop found two stacking improvements that drove RMS from
**8.27 → 0.546 px** (15x). Neither is "sub-pixel refinement" in the way
the question was framed:

1. **Board geometry was wrong** (8.27 → 1.44 px, 5.7x). The YAML claimed
   `tag_spacing = 0.30` (15mm gap), but global multi-frame fit shows the
   actual board has spacing ≈ 0.54 (~27mm gap). A second entry
   `empirical_10x7_s054` was added to `config/apriltag_board.yaml`; the
   original is preserved for any other recording that genuinely uses the
   15mm-gap board. **Verify which board your other recordings use** before
   re-running their calibrations.
2. **Detection-time upscale of the input image** (1.44 → 0.546 px, 2.6x).
   cv2.aruco's quad detector is pixel-resolution-limited on this unwrap
   (1280×1280 from a 1920×1920 source). Bicubic-upscaling the unwrap 8x
   before running the detector gives the quad-localizer more sub-pixel
   resolution to lock onto. *Built-in* refinement modes (SUBPIX, CONTOUR,
   APRILTAG) did **not** help meaningfully — all clustered at 1.42-1.46 px
   on the unscaled unwrap.

State machine for the overnight loop. Each iteration picks the top `[ ]`
item, implements it as a method in `bench_subpixel.py`, runs the benchmark,
appends a row to `bench_subpixel.jsonl`, then marks the item `[x]`.

**Target dataset:** `data/insta360_calibration/lens0_combined.mp4` (1920x1920
raw fisheye, 22102 frames, ~12 min). The benchmark equidistant-unwraps each
frame to 1280x1280 @ 110-deg FOV before detection, then pinhole-calibrates the
unwrapped views (mirrors `two_stage_calibrate.py`). Methods vary only the
detect-and-refine step; everything else (unwrap, calibrateCamera flags,
corner-shift=2) is fixed.

**Stop conditions:** an iteration achieves RMS < 1.0 px, OR the queue is empty.

## Status
- Best RMS so far: **see `bench_subpixel.jsonl`**
- Last iteration: (running) baseline re-run on empirical board geometry

## **CRITICAL FINDING (2026-05-15 evening)** — sub-pixel isn't the bottleneck

Per-tag PnP on lens0_combined.mp4 (single tag, 4 corners) gives ~0.4 px error.
Pooled multi-tag PnP gives 8-10 px — the **board geometry in
`apriltag_board.yaml` was wrong** (tag_spacing=0.30 → actual ~0.54).

Global fit (`debug_global_spacing.py`) on 11 frames: optimal isotropic
spacing 0.54 (~27mm gap), anisotropic sx=0.525, sy=0.557. Even at the
optimum, residual is ~4 px — the regular-grid model still has ~4 px of
slop, presumably from non-pinhole equidistant-unwrap distortion that the
calibrateCamera step will absorb into K, D.

A second YAML entry `empirical_10x7_s054` has been added and is what
`bench_subpixel.py` now uses. The original `calib_io_10x7` is preserved
for any other recording that uses the actual 15mm-gap board.

**Implication for the queue:** sub-pixel refinement can plausibly improve
from ~4 px to ~2 px, but probably not below ~1 px until the residual
board-geometry slop is also addressed (e.g. by Kalibr-style bundle
adjustment over the tag positions). Re-evaluate after the corrected
baseline lands.

## Final results (sorted by RMS, latest entry per method/unwrap-size)

```
method                          unwrap     RMS    views   med    p95
upscale8x_norefine              1280x1280  0.546  246    0.56   0.85   ← BEST
upscale4x_norefine              1280x1280  0.878  580    0.87   1.35
upscale4x_norefine              1920x1920  0.907  407    0.92   1.48   (1920 unwrap was WORSE)
upscale2x_norefine              1280x1280  1.160  527    1.11   1.70
upscale2x_subpix                1280x1280  1.167  527    1.11   1.71
aruco_norefine                  1280x1280  1.423  122    1.34   1.88
aruco_subpix_w7_tight           1280x1280  1.441  122    1.36   1.85
aruco_subpix_w11                1280x1280  1.442  122    1.36   1.84
aruco_subpix_w7                 1280x1280  1.442  122    1.36   1.84
baseline_aruco_subpix           1280x1280  1.442  122    1.36   1.86
aruco_subpix_w3                 1280x1280  1.444  122    1.36   1.92
aruco_contour                   1280x1280  1.461  122    1.34   2.17
upscale2x_norefine              1920x1920  1.458  621    1.41   2.25
clahe_then_subpix               1280x1280  1.490   74    1.35   2.18

(pre-fix baseline_aruco_subpix at YAML spacing=0.30: RMS=8.27 px)
```

## Methods queue (mostly done — kept as historical record)

### Aruco built-in refinement modes
- [x] `baseline_aruco_subpix` — RMS 1.442 px (corrected board), 8.268 (broken board)
- [x] `aruco_apriltag_refine` — **broken on OpenCV 4.10** (0 markers; cf
  opencv issues #2643, #23437); skipped
- [x] `aruco_contour` — 1.461 px (CONTOUR regression mentioned in #23437 may
  affect this; comparable to SUBPIX here)
- [x] `aruco_norefine` — 1.423 px (no refinement = ~ same as any refinement)

### Aruco sub-pixel param sweep
- [x] `aruco_subpix_w3` / `w5` / `w7` / `w11` — all 1.44 ± 0.005 px; window size
  doesn't matter on this dataset
- [x] `aruco_subpix_w7_tight` — max_iter=100, min_acc=0.001; no improvement

### Image preprocessing before detection
- [x] `clahe_then_subpix` — 1.490 px (slightly WORSE; CLAHE knocks down view
  count from 122 to 74)
- [x] `upscale2x_norefine` — 1.160 px (first real improvement)
- [x] `upscale2x_subpix` — 1.167 px (subpix on upscaled doesn't help vs norefine)
- [x] `upscale4x_norefine` — 0.878 px (under 1!)
- [x] `upscale8x_norefine` — 0.546 px (current best)
- [x] Higher-res unwrap (1920×1920) at upscale 2x and 4x — slightly WORSE than
  1280 unwrap (the 1920 unwrap is undersampling at periphery — for a
  110-deg FOV unwrap, the fisheye→pinhole magnification at periphery is
  high and 1920 doesn't have enough source pixels)

### External AprilTag libraries
- [x] `pupil_apriltags_default` — **broken on this dataset** (detects 0-3
  tags on close-up frames where aruco gets 18; works on synthetic;
  detection mechanism rejects the unwrapped views for an unclear reason).
  Skipped.

### Ideas not yet tried (worth a shot if pushing further)

- [ ] **upscale16x** — push the trend; expected diminishing returns; memory
  ~1.6 GB grayscale per frame at 1280×16
- [ ] **Outlier filter then re-calibrate** — drop top 5-10% per-view RMS,
  re-run calibrateCamera. On a 0.546-px baseline, this could shave another
  ~10-20%
- [ ] **Bundle-adjust tag positions** — the remaining ~0.55 px residual is
  partly board geometry slop (per-tag PnP is 0.4 px, so almost-but-not-all
  of the residual is in inter-tag positioning). Kalibr does this.
- [ ] **Stereographic unwrap** instead of equidistant — preserves angles
  better; might give the detector cleaner corners. Would need new
  Rectifier mode.
- [ ] **Per-frame motion-blur filter** — reject views where the gradient
  magnitude is below a threshold; might help on close-up shots
- [ ] **Calibrate directly on raw fisheye** (`cv2.fisheye.calibrate`) with
  upscale8x — eliminates the unwrap interpolation entirely. The previous
  attempt with fisheye calibrate diverged at the default detector, but
  with upscaled+norefine corners it might converge now.

## How to update this file
- Mark `[x]` when an item finishes (success or failure both count).
- Add new ideas at the bottom as they come up.
- If something achieves RMS < 1.0 px: stop the loop and leave a clear note here.

## References — consult these when stuck or to pick the next idea

Always use WebFetch / WebSearch on these before improvising:

- **OpenCV ArUco docs** — refinement methods, DetectorParameters fields, what
  `cornerRefinementMethod`, `cornerRefinementWinSize`,
  `cornerRefinementMaxIterations`, `cornerRefinementMinAccuracy`,
  `aprilTagDeglitch`, `aprilTagQuadDecimate`, etc. actually do.
  https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html
  https://docs.opencv.org/4.x/d2/d1a/classcv_1_1aruco_1_1DetectorParameters.html
- **OpenCV cornerSubPix** — TermCriteria semantics, window/zeroZone trade-offs.
  https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html
- **OpenCV calibrateCamera flags** — CALIB_RATIONAL_MODEL, CALIB_THIN_PRISM_MODEL,
  CALIB_TILTED_MODEL, CALIB_USE_LU, etc.
  https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
- **Kalibr** — how the canonical AprilGrid pipeline does corner refinement and
  bundle adjustment. Look at
  `aslam_cv_python/python/sm/aprilgrid_target.py` and the
  `aslam_offline_calibration/kalibr` calibrator entry point.
  https://github.com/ethz-asl/kalibr
- **ai4ce/insta360_ros_driver** — driver for the same camera family; check
  whether they ship intrinsics or document the unwarp projection used by the
  app vs SDK.
  https://github.com/ai4ce/insta360_ros_driver
- **pupil-labs/apriltags** — the Python binding for the official AprilTag 3
  detector. Often gives 0.1-0.3 px corner accuracy out of the box, much better
  than cv2.aruco's. `pip install pupil-apriltags`.
  https://github.com/pupil-labs/apriltags
- **AprilTag 3 paper (Wang & Olson 2016)** — for understanding what
  edge-refinement does and why it dominates corner accuracy.
  https://april.eecs.umich.edu/papers/details.php?name=wang2016iros
- **OpenCV PR / issue archive** — search GitHub issues for known sub-pixel
  bugs in `CORNER_REFINE_APRILTAG` (there have been a few). gh issue list -R
  opencv/opencv --search "aruco refine".

Logging discipline: when you consult a reference, jot the key takeaway as a
one-liner in the iteration's JSONL row under `"notes"`. Future iterations
(and the morning review) shouldn't have to re-derive what you learned.
