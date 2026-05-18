"""Camera-pose estimation from per-frame detections.

Each module is a ``PoseEstimator`` implementation (per the protocol in
:mod:`core.pipeline`):

  - ``known_board``    -- AprilGrid with a known layout from config;
                          pooled PnP across all detected tags per frame
  - ``learned_layout`` -- independent ArUco markers; the inter-marker
                          layout is learned from co-visibility in a
                          first pass, then the camera is tracked via
                          pooled PnP in a second pass
"""
