import sys

import cv2
import numpy as np

from detect_dots import detect_black_circular_dots


def compute_hinge_angle(frame):
    detections, _ = detect_black_circular_dots(frame)
    if len(detections) != 4:
        raise ValueError(f"expected 4 dots, found {len(detections)}")

    centers = np.array([d["center"] for d in detections])

    mid_x = frame.shape[1] / 2.0
    left = centers[centers[:, 0] < mid_x]
    right = centers[centers[:, 0] >= mid_x]
    if len(left) != 2 or len(right) != 2:
        raise ValueError(
            f"split failed: {len(left)} left, {len(right)} right"
        )

    left_top, left_bottom = left[np.argsort(left[:, 1])]
    right_top, right_bottom = right[np.argsort(right[:, 1])]

    a = left_bottom - left_top
    b = right_bottom - right_top

    cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))

    return np.degrees(np.arccos(cos_theta)), {
        "left_top": left_top,
        "left_bottom": left_bottom,
        "right_top": right_top,
        "right_bottom": right_bottom,
    }


if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else "four_dots_test.png"
    frame = cv2.imread(img_path)
    if frame is None:
        raise SystemExit(f"could not read {img_path}")

    angle, pts = compute_hinge_angle(frame)
    for name, p in pts.items():
        print(f"  {name}: ({p[0]:.2f}, {p[1]:.2f})")
    print(f"hinge angle: {angle:.4f} deg")
