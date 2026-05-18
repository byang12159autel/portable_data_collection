"""Camera calibration.

Each module is a ``Calibrator`` implementation (per the ``Calibrator``
protocol in :mod:`core.pipeline`):

  - ``fisheye``     -- ``cv2.fisheye.calibrate`` (the primary path)
  - ``two_stage``   -- equidistant unwrap + pinhole refine (fallback when
                       fisheye won't converge)
  - ``pinhole``     -- standard ``cv2.calibrateCamera`` for already-unwarped
                       inputs (e.g. Insta360 Studio single-lens exports)
  - ``auto``        -- sweeps thresholds, picks first acceptable result

The ``Rectifier`` implementation that consumes these calibrations lives
in :mod:`core.rectify`.
"""
