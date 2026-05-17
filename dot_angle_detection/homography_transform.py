import cv2
import numpy as np


def homography_from_aruco_pose(K, rvec, tvec):
    """
    Build plane-to-image and image-to-plane homographies from an ArUco pose.

    OpenCV ArUco pose convention:
        P_camera = R * P_marker + t

    Marker plane:
        Z_marker = 0

    Homography:
        s [u, v, 1]^T = K [r1 r2 t] [X, Y, 1]^T

    Args:
        K:    3x3 camera intrinsic matrix
        rvec: 3x1 rotation vector from ArUco pose estimation
        tvec: 3x1 translation vector from ArUco pose estimation

    Returns:
        H_plane_to_img: 3x3 homography from marker plane coordinates to image pixels
        H_img_to_plane: 3x3 homography from image pixels to marker plane coordinates
    """

    R, _ = cv2.Rodrigues(rvec)
    t = np.asarray(tvec, dtype=np.float64).reshape(3, 1)

    r1 = R[:, 0:1]
    r2 = R[:, 1:2]

    H_plane_to_img = K @ np.hstack([r1, r2, t])
    H_img_to_plane = np.linalg.inv(H_plane_to_img)

    return H_plane_to_img, H_img_to_plane


def apply_homography(H, point_uv):
    """
    Apply homography to a single 2D point.

    Args:
        H: 3x3 homography
        point_uv: image point, [u, v]

    Returns:
        2D transformed point [x, y]
    """

    u, v = point_uv
    p = np.array([u, v, 1.0], dtype=np.float64)

    q = H @ p
    q = q / q[2]

    return q[:2]