"""General camera I/O helpers.

  - ``convert``     -- demux an Insta360 ``.insv`` into per-lens ``.mp4`` streams
  - ``inspect``     -- ``ffprobe`` summary + frame-0 dump for any video container
  - ``split``       -- side-by-side splitter for the live USB dual-fisheye stream

Frame iteration is currently inlined in each app via ``cv2.VideoCapture``;
add a ``FrameSource`` implementation here if/when the live and offline
paths grow enough divergence to warrant the abstraction.
"""
