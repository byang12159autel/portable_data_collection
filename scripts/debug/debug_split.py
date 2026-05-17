import os
import cv2

from pose_calibration.camera.split import split_dual_fisheye

path = "data/VID_20260515_121341_00_001.insv"
cap = cv2.VideoCapture(path)
print("opened:", cap.isOpened())
print("frame_count:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
print(
    "reported size:",
    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    "x",
    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
)
ok, bgr = cap.read()
cap.release()
print("read ok:", ok)
if ok:
    print("actual shape (H, W, C):", bgr.shape)
    front, back = split_dual_fisheye(bgr)
    print("front shape:", front.shape, "back shape:", back.shape)
    for name, img in [("source", bgr), ("front", front), ("back", back)]:
        out = f"data/debug_{name}.png"
        ok2 = cv2.imwrite(out, img)
        size = os.path.getsize(out) if os.path.exists(out) else "missing"
        print(f"  {out}: write_ok={ok2}, size={size}")
