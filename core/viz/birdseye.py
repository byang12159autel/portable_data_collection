"""Top-down warp of the marker plane + helpers for embedding it as an inset.

The base ``birdseye`` warps an image into a metric top-down view of the
marker plane: origin at the centre, +X right, +Y up, pixels-per-metre
identical on both axes (so the view isn't aspect-stretched). The
``embed_inset`` helper drops a smaller version into the bottom-right
corner of another canvas.
"""

from __future__ import annotations

import cv2
import numpy as np


def birdseye(img: np.ndarray, H_plane_to_img: np.ndarray,
             extent_m: float, width_px: int,
             height_px: int | None = None) -> np.ndarray:
    """Warp the image into a top-down view of the marker plane.

    The plane origin maps to the center of the output, +X right, +Y up.
    ``extent_m`` is the *x* half-extent; the y half-extent is derived
    from ``height_px`` so the pixels-per-meter scale matches on both
    axes (no aspect-ratio distortion).
    """
    if height_px is None:
        height_px = width_px
    s = width_px / (2.0 * extent_m)            # pixels per metre
    extent_y_m = height_px / (2.0 * s)         # half-extent on Y
    H_plane_to_out = np.array(
        [[s, 0.0, width_px / 2.0],
         [0.0, -s, height_px / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    H_img_to_out = H_plane_to_out @ np.linalg.inv(H_plane_to_img)
    out = cv2.warpPerspective(
        img, H_img_to_out, (width_px, height_px),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
    )
    cx, cy = width_px // 2, height_px // 2
    cv2.line(out, (0, cy), (width_px, cy), (255, 200, 80), 1)
    cv2.line(out, (cx, 0), (cx, height_px), (255, 200, 80), 1)
    cv2.arrowedLine(out, (cx, cy), (cx + int(s * extent_m * 0.5), cy),
                    (0, 0, 255), 2, tipLength=0.15)
    cv2.arrowedLine(out, (cx, cy), (cx, cy - int(s * extent_y_m * 0.5)),
                    (0, 255, 0), 2, tipLength=0.15)
    cv2.putText(out,
                f"birds-eye  x+-{extent_m * 1000:.0f}mm  y+-{extent_y_m * 1000:.0f}mm",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def embed_inset(canvas: np.ndarray, inset: np.ndarray, margin: int = 12) -> None:
    """Paste ``inset`` into the bottom-right corner of ``canvas`` with a border."""
    h, w = canvas.shape[:2]
    ih, iw = inset.shape[:2]
    y0 = h - ih - margin
    x0 = w - iw - margin
    canvas[y0:y0 + ih, x0:x0 + iw] = inset
    cv2.rectangle(canvas, (x0 - 1, y0 - 1), (x0 + iw, y0 + ih),
                  (255, 255, 255), 1)
