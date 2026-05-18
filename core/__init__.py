"""Shared infrastructure used by every component.

Three components — ``calibration``, ``pose_6d``, ``gripper`` — each import
from here but never from each other. Anything that ends up needed by more
than one of them lives in ``core``:

  - ``pipeline``   -- stage protocols + ``DataPipeline``/``RigPipeline``
  - ``rectify``    -- Stage-1 (equidistant unwrap) + Stage-2 (pinhole
                      undistort) ``Rectifier``
  - ``markers``    -- ArUco / ChArUco / AprilTag / AprilGrid detection
                      + YAML config loaders
  - ``geometry``   -- ``homography_from_aruco_pose`` and friends
  - ``camera``     -- ``.insv`` demux, ``ffprobe`` inspection, side-by-side
                      splitting of the live dual-fisheye stream
  - ``viz``        -- shared drawing helpers (axes, plane grid, bird's-eye,
                      viser panel boilerplate)
"""
