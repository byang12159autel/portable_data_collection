import cv2
import numpy as np


def detect_black_circular_dots(
    frame,
    K=None,
    D=None,
    threshold=80,
    min_area=20,
    max_area=5000,
    min_circularity=0.55,
):
    # 1. Undistort
    if K is not None and D is not None:
        frame = cv2.undistort(frame, K, D)

    # 2. Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 3. Slight blur
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 4. Threshold black dots
    _, mask = cv2.threshold(
        gray_blur,
        threshold,
        255,
        cv2.THRESH_BINARY_INV,
    )

    # 5. Morphology cleanup
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 6. Contours
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    detections = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(c, True)
        if perimeter <= 1e-6:
            continue

        circularity = 4.0 * np.pi * area / (perimeter * perimeter)

        if circularity < min_circularity:
            continue

        if len(c) >= 5:
            ellipse = cv2.fitEllipse(c)
            center = np.array(ellipse[0], dtype=float)
            axes = ellipse[1]
            ellipse_angle = ellipse[2]
        else:
            M = cv2.moments(c)
            if abs(M["m00"]) < 1e-9:
                continue
            center = np.array(
                [M["m10"] / M["m00"], M["m01"] / M["m00"]],
                dtype=float,
            )
            axes = None
            ellipse_angle = None

        detections.append({
            "center": center,
            "area": area,
            "circularity": circularity,
            "axes": axes,
            "ellipse_angle": ellipse_angle,
            "contour": c,
        })

    return detections, mask


def draw_detections(frame, detections):
    out = frame.copy()
    for i, det in enumerate(detections):
        cx, cy = det["center"]
        cv2.drawContours(out, [det["contour"]], -1, (0, 255, 0), 1)
        cv2.circle(out, (int(round(cx)), int(round(cy))), 6, (0, 0, 255), 1)
        cv2.drawMarker(
            out,
            (int(round(cx)), int(round(cy))),
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=10,
            thickness=1,
        )
        label = f"#{i} ({cx:.1f},{cy:.1f}) c={det['circularity']:.2f}"
        cv2.putText(
            out,
            label,
            (int(round(cx)) + 8, int(round(cy)) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return out


if __name__ == "__main__":
    import sys

    img_path = sys.argv[1] if len(sys.argv) > 1 else "four_dots_test.png"
    frame = cv2.imread(img_path)
    if frame is None:
        raise SystemExit(f"could not read {img_path}")

    detections, mask = detect_black_circular_dots(frame)

    print(f"found {len(detections)} dot(s)")
    for i, det in enumerate(detections):
        cx, cy = det["center"]
        print(
            f"  #{i}: center=({cx:.2f}, {cy:.2f}) "
            f"area={det['area']:.1f} circ={det['circularity']:.3f} "
            f"axes={det['axes']} angle={det['ellipse_angle']}"
        )

    vis = draw_detections(frame, detections)
    cv2.imwrite("four_dots_test_detections.png", vis)
    cv2.imwrite("four_dots_test_mask.png", mask)
    print("wrote four_dots_test_detections.png and four_dots_test_mask.png")
