"""Gripper-state estimation (component 3).

Per frame, a marker-anchored homography lifts black/white dot detections
into the marker plane and computes the chopstick hinge angle there.

  - ``dots``      -- black/white circular dot detection
  - ``hinge``     -- plane-space hinge-angle math (no I/O) [TODO: split]
  - ``pipeline``  -- end-to-end CLI: insv -> rectify -> ArUco pose ->
                     plane-space dots -> hinge angle -> annotated mp4 +
                     optional viser preview
  - ``tune``      -- interactive dot-area threshold tuner
"""
