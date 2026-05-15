"""Offline Insta360 dual-fisheye processing for marker-based pose work.

Pipeline:

    .insv  --convert.py-->  dual_fisheye.mp4
                                  |
                                  v
    board recording --calibrate.py--> intrinsics.npz  (K, D per lens)
                                  |
                                  v
                            rectify.py / replay_insta.py
                            (split + fisheye -> pinhole + detect)
"""
