"""Camera calibration + rectification.

Each module is a ``Calibrator`` implementation (per the ``Calibrator``
protocol in :mod:`pose_calibration.pipeline`):

  - ``fisheye``     -- ``cv2.fisheye.calibrate`` (the primary path)
  - ``two_stage``   -- equidistant unwrap + pinhole refine (fallback when
                       fisheye won't converge)
  - ``pinhole``     -- standard ``cv2.calibrateCamera`` for already-unwarped
                       inputs (e.g. Insta360 Studio single-lens exports)
  - ``auto``        -- sweeps thresholds, picks first acceptable result

``rectify`` implements the ``Rectifier`` protocol on top of any of these
calibration outputs.
"""
