"""Globally fit tag_size and tag_spacing across many frames.

For each candidate (tag_size, tag_spacing): for every frame with >=8 tags
detected, run pooled solvePnP (zero distortion, K=Ktest), sum per-corner
errors, divide by total corner count. Minimum over the candidate-grid is
the best-fit board geometry — the value that should be in
config/apriltag_board.yaml.

Per-tag PnP already shows the corner detector is accurate (~0.4 px); this
script just verifies the geometric model.
"""
import cv2, math, numpy as np
from pathlib import Path
import dataclasses
from scipy.optimize import minimize_scalar, minimize

from pose_calibration.detect_marker import (
    detect_aruco_markers, load_apriltag_grid_configs,
)
from pose_calibration.insta360.rectify import Rectifier

VIDEO = 'data/insta360_calibration/lens0_combined.mp4'
FRAME_STRIDE = 60  # collect ~370 frames

grids = load_apriltag_grid_configs(Path('config/apriltag_board.yaml'))
_, g = next(iter(grids.items()))

cap = cv2.VideoCapture(VIDEO)
n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f'collecting from {VIDEO} ({n_total} frames, stride={FRAME_STRIDE})')

rect = None
frames_views: list[tuple[np.ndarray, np.ndarray]] = []  # (img_pts (N,2), tag_ids (N,))
idx = -1
while True:
    ok, bgr = cap.read()
    if not ok: break
    idx += 1
    if idx % FRAME_STRIDE != 0: continue
    if rect is None:
        fh, fw = bgr.shape[:2]
        f_eq = fw / math.pi
        K = np.array([[f_eq,0,fw/2],[0,f_eq,fh/2],[0,0,1]], dtype=np.float64)
        rect = Rectifier.build(K, np.zeros(4), (1280, 1280), 110.0)
    rough = rect.apply(bgr)
    rgb = cv2.cvtColor(rough, cv2.COLOR_BGR2RGB)
    c, ids = detect_aruco_markers(rgb, marker_dict=g.cv2_dictionary, allowed_ids=set(g.tag_ids))
    if ids is None or len(ids) < 8: continue
    img_pts_list, tag_id_list = [], []
    for cc, tid in zip(c, ids.flatten()):
        for k in range(4):
            img_pts_list.append(cc.reshape(4,2)[k])
            tag_id_list.append((int(tid), k))
    frames_views.append((np.array(img_pts_list, dtype=np.float32), tag_id_list))
cap.release()
print(f'collected {len(frames_views)} usable frames')

Ktest = np.array([[800,0,640],[0,800,640],[0,0,1]], dtype=np.float64)

def build_obj(sx, sy, ts=0.050, cols=10, rows=7):
    sx_m = ts * (1.0 + sx)
    sy_m = ts * (1.0 + sy)
    obj = {}
    for r in range(rows):
        for col in range(cols):
            tid = r * cols + col
            x0, y0 = col * sx_m, r * sy_m
            obj[tid] = [
                np.array([x0,           y0 + ts, 0.0], dtype=np.float32),
                np.array([x0 + ts,      y0 + ts, 0.0], dtype=np.float32),
                np.array([x0 + ts,      y0,      0.0], dtype=np.float32),
                np.array([x0,           y0,      0.0], dtype=np.float32),
            ]
    return obj

def total_err(params):
    sx, sy = params[0], params[1]
    obj_dict = build_obj(sx, sy)
    total = 0.0
    n = 0
    for img_pts, tag_id_list in frames_views:
        # Build (obj_pts, img_pts) for this frame, applying corner shift=2 per tag
        obj_pts = []
        for tid, k in tag_id_list:
            k_shifted = (k + 2) % 4
            obj_pts.append(obj_dict[tid][k_shifted])
        obj_pts = np.array(obj_pts, dtype=np.float32)
        ok, r, t = cv2.solvePnP(obj_pts, img_pts, Ktest, np.zeros(5), flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok: continue
        rep, _ = cv2.projectPoints(obj_pts, r, t, Ktest, np.zeros(5))
        errs = np.linalg.norm(rep.reshape(-1,2) - img_pts, axis=1)
        total += float(errs.sum())
        n += len(errs)
    return total / max(n, 1)

# 1D sweep first
print('\n1D sweep of isotropic spacing:')
for s in np.arange(0.0, 1.6, 0.1):
    print(f'  spacing={s:.2f}: mean_err = {total_err([s, s]):.3f} px')

# Now 2D refine
res = minimize(total_err, x0=[0.515, 0.515], method='Nelder-Mead', options={'xatol':1e-5, 'fatol':1e-5})
print(f'\noptimal sx={res.x[0]:.4f}, sy={res.x[1]:.4f}, mean_err={res.fun:.4f} px')

# What does this say about the board?
print(f'\nimplied stride_x = tag_size * (1+sx) = 0.050 * {1+res.x[0]:.4f} = {0.050*(1+res.x[0])*1000:.2f} mm')
print(f'implied stride_y = tag_size * (1+sy) = 0.050 * {1+res.x[1]:.4f} = {0.050*(1+res.x[1])*1000:.2f} mm')
print(f'implied gap_x = stride_x - tag_size = {(0.050*(1+res.x[0]) - 0.050)*1000:.2f} mm')
print(f'implied gap_y = stride_y - tag_size = {(0.050*(1+res.x[1]) - 0.050)*1000:.2f} mm')
