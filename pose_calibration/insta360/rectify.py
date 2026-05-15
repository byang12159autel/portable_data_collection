#!/usr/bin/env python3
"""Fisheye -> single-pinhole rectification per Insta360 lens.

Build a per-lens remap LUT once from the calibration file, apply it to
every frame of the corresponding per-lens video. Each rectified frame
behaves like a standard pinhole camera with the synthetic intrinsics
``K_pinhole`` returned here, so downstream ``cv2.solvePnP`` works
directly (with zero distortion).

Single-pinhole-per-lens is a conscious simplification — see the README
for the multi-virtual-pinhole upgrade path that recovers the full 360°
FOV the camera actually offers.

CLI mode writes one rectified ``.mp4`` per lens plus
``pinhole_intrinsics.npz`` for downstream pose code::

    pixi run python -m pose_calibration.insta360.rectify \\
        --front-video data/scene_lens0.mp4 \\
        --back-video data/scene_lens1.mp4 \\
        --intrinsics data/insta360_intrinsics.npz \\
        --output-dir data/rectified \\
        --fov-deg 110 --out-width 1280 --out-height 1280
"""

from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path  # noqa: TC003 — tyro needs Path at runtime

import cv2
import numpy as np
import tyro


# ---------------------------------------------------------------------------
# Per-lens rectifier
# ---------------------------------------------------------------------------


def _pinhole_K(out_width: int, out_height: int, fov_deg: float) -> np.ndarray:
    """Synthetic pinhole intrinsics that span ``fov_deg`` horizontally."""
    fov_rad = math.radians(fov_deg)
    fx = (out_width / 2.0) / math.tan(fov_rad / 2.0)
    return np.array(
        [[fx, 0.0, out_width / 2.0],
         [0.0, fx, out_height / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


@dataclasses.dataclass
class Rectifier:
    """Precomputed fisheye -> pinhole remap for one lens.

    For single-stage (fisheye-only) calibration the map is built directly
    with ``cv2.fisheye.initUndistortRectifyMap``. For two-stage
    calibrations (equidistant unwrap + pinhole refine), ``build_two_stage``
    composes the stage 1 and stage 2 maps into a single LUT so the
    downstream code path is identical — one ``cv2.remap`` per frame.
    """

    K_pinhole: np.ndarray
    map1: np.ndarray
    map2: np.ndarray
    out_size: tuple[int, int]  # (W, H)

    @classmethod
    def build(
        cls,
        K_fisheye: np.ndarray,
        D_fisheye: np.ndarray,
        out_size: tuple[int, int],
        fov_deg: float,
    ) -> "Rectifier":
        K_p = _pinhole_K(out_size[0], out_size[1], fov_deg)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K_fisheye.astype(np.float64),
            D_fisheye.astype(np.float64).reshape(4, 1),
            np.eye(3, dtype=np.float64),
            K_p,
            out_size,
            cv2.CV_16SC2,
        )
        return cls(K_pinhole=K_p, map1=map1, map2=map2, out_size=out_size)

    @classmethod
    def build_two_stage(
        cls,
        K_fisheye: np.ndarray,
        D_fisheye: np.ndarray,
        K_pinhole_rough: np.ndarray,
        K_pinhole_refined: np.ndarray,
        D_pinhole_refined: np.ndarray,
        pinhole_size: tuple[int, int],
    ) -> "Rectifier":
        """Compose stage 1 (fisheye -> rough pinhole) and stage 2 (rough -> final).

        Output K is ``K_pinhole_rough`` — same as what ``two_stage_calibrate``
        used at calibration time. Downstream solvePnP should use this K
        with zero distortion.
        """
        # Stage 1: fisheye -> rough pinhole. Float maps so we can sample
        # them at non-integer (stage 2) positions.
        s1_x, s1_y = cv2.fisheye.initUndistortRectifyMap(
            K_fisheye.astype(np.float64),
            D_fisheye.astype(np.float64).reshape(4, 1),
            np.eye(3, dtype=np.float64),
            K_pinhole_rough.astype(np.float64),
            pinhole_size,
            cv2.CV_32FC1,
        )
        # Stage 2: rough pinhole -> final pinhole at the same K (so the
        # final image behaves like a clean pinhole with K_pinhole_rough).
        s2_x, s2_y = cv2.initUndistortRectifyMap(
            K_pinhole_refined.astype(np.float64),
            D_pinhole_refined.astype(np.float64),
            np.eye(3, dtype=np.float64),
            K_pinhole_rough.astype(np.float64),
            pinhole_size,
            cv2.CV_32FC1,
        )
        # Compose by sampling stage 1's map at stage 2's positions.
        composed_x = cv2.remap(s1_x, s2_x, s2_y, cv2.INTER_LINEAR)
        composed_y = cv2.remap(s1_y, s2_x, s2_y, cv2.INTER_LINEAR)
        # Pack into the fixed-point format remap prefers.
        fixed_xy, fixed_extra = cv2.convertMaps(
            composed_x, composed_y, cv2.CV_16SC2,
        )
        return cls(
            K_pinhole=K_pinhole_rough.astype(np.float64),
            map1=fixed_xy,
            map2=fixed_extra,
            out_size=pinhole_size,
        )

    def apply(self, lens_frame: np.ndarray) -> np.ndarray:
        return cv2.remap(
            lens_frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )


# ---------------------------------------------------------------------------
# Intrinsics bundle (mirrors the .npz layout from calibrate.py)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _TwoStageData:
    """Extra fields from ``two_stage_calibrate.py`` output, per lens."""

    K_pinhole_rough: np.ndarray
    K_pinhole_refined: np.ndarray
    D_pinhole_refined: np.ndarray
    pinhole_size: tuple[int, int]


@dataclasses.dataclass
class IntrinsicsBundle:
    """Per-lens intrinsics loaded from calibrate.py / two_stage_calibrate.py output.

    Either lens may be missing if it wasn't calibrated yet — fields are
    ``None`` in that case. ``image_size`` is always present once any lens
    has been calibrated. ``two_stage_front`` / ``two_stage_back`` are
    populated only when the .npz came from ``two_stage_calibrate``.
    """

    K_front: np.ndarray | None
    D_front: np.ndarray | None
    K_back: np.ndarray | None
    D_back: np.ndarray | None
    image_size: tuple[int, int]

    two_stage_front: _TwoStageData | None = None
    two_stage_back: _TwoStageData | None = None

    @classmethod
    def load(cls, path: Path) -> "IntrinsicsBundle":
        d = np.load(str(path))
        size = tuple(int(v) for v in d["image_size"])
        if len(size) != 2:
            raise ValueError(f"image_size in {path} has unexpected shape {size}")

        def _two_stage(label: str) -> _TwoStageData | None:
            keys = (
                f"K_{label}_pinhole_rough",
                f"K_{label}_pinhole_refined",
                f"D_{label}_pinhole_refined",
                f"pinhole_size_{label}",
            )
            if not all(k in d.files for k in keys):
                return None
            pinhole_size = tuple(int(v) for v in d[f"pinhole_size_{label}"])
            return _TwoStageData(
                K_pinhole_rough=d[f"K_{label}_pinhole_rough"],
                K_pinhole_refined=d[f"K_{label}_pinhole_refined"],
                D_pinhole_refined=d[f"D_{label}_pinhole_refined"],
                pinhole_size=pinhole_size,  # type: ignore[arg-type]
            )

        return cls(
            K_front=d["K_front"] if "K_front" in d.files else None,
            D_front=d["D_front"] if "D_front" in d.files else None,
            K_back=d["K_back"] if "K_back" in d.files else None,
            D_back=d["D_back"] if "D_back" in d.files else None,
            image_size=size,  # type: ignore[arg-type]
            two_stage_front=_two_stage("front"),
            two_stage_back=_two_stage("back"),
        )

    def rectifier_for(
        self,
        lens: str,
        out_size: tuple[int, int],
        fov_deg: float,
    ) -> Rectifier:
        K = getattr(self, f"K_{lens}")
        D = getattr(self, f"D_{lens}")
        if K is None or D is None:
            raise RuntimeError(
                f"Intrinsics for '{lens}' lens not present in calibration file"
            )

        two_stage: _TwoStageData | None = getattr(self, f"two_stage_{lens}")
        if two_stage is not None:
            if out_size != two_stage.pinhole_size:
                print(
                    f"  [{lens}] two-stage calibration baked in pinhole size "
                    f"{two_stage.pinhole_size}, overriding requested {out_size}"
                )
            return Rectifier.build_two_stage(
                K_fisheye=K,
                D_fisheye=D,
                K_pinhole_rough=two_stage.K_pinhole_rough,
                K_pinhole_refined=two_stage.K_pinhole_refined,
                D_pinhole_refined=two_stage.D_pinhole_refined,
                pinhole_size=two_stage.pinhole_size,
            )

        return Rectifier.build(K, D, out_size, fov_deg)


# ---------------------------------------------------------------------------
# CLI: write rectified per-lens mp4s
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Args:
    """Write rectified mp4(s) + pinhole intrinsics for the provided lens videos."""

    intrinsics: Path
    """Per-lens calibration .npz from calibrate.py."""

    output_dir: Path
    """Directory for front_rectified.mp4 / back_rectified.mp4 + pinhole_intrinsics.npz."""

    front_video: Path | None = None
    """Per-lens recording of the front lens. Optional; skipped if not given."""

    back_video: Path | None = None
    """Per-lens recording of the back lens. Optional; skipped if not given."""

    fov_deg: float = 110.0
    """Horizontal FOV of the virtual pinhole, degrees. 90-120 is typical."""

    out_width: int = 1280
    """Output width in pixels."""

    out_height: int = 1280
    """Output height in pixels. Keep square unless you have a reason."""

    fourcc: str = "mp4v"
    """OpenCV VideoWriter fourcc."""


def _rectify_one(
    lens: str,
    video: Path,
    rectifier: Rectifier,
    out_path: Path,
    fourcc: str,
) -> int:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[{lens}] {video}  ({n_total} frames @ {fps:.1f} FPS) -> {out_path}")

    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*fourcc), fps, rectifier.out_size
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"VideoWriter failed with fourcc='{fourcc}'")

    n = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        writer.write(rectifier.apply(bgr))
        n += 1
        if n % 60 == 0:
            print(f"[{lens}]  rectified {n}/{n_total}")
    cap.release()
    writer.release()
    return n


def main(args: Args) -> None:
    if args.front_video is None and args.back_video is None:
        print("Error: at least one of --front-video / --back-video is required", file=sys.stderr)
        sys.exit(1)
    if not args.intrinsics.is_file():
        print(f"Error: intrinsics not found: {args.intrinsics}", file=sys.stderr)
        sys.exit(1)

    bundle = IntrinsicsBundle.load(args.intrinsics)
    out_size = (args.out_width, args.out_height)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Source size {bundle.image_size[0]}x{bundle.image_size[1]} -> "
        f"pinhole {out_size[0]}x{out_size[1]} @ {args.fov_deg:.1f} deg FOV"
    )

    rectifiers: dict[str, Rectifier] = {}
    for lens, video in [("front", args.front_video), ("back", args.back_video)]:
        if video is None:
            continue
        rectifier = bundle.rectifier_for(lens, out_size, args.fov_deg)
        rectifiers[lens] = rectifier
        out_path = args.output_dir / f"{lens}_rectified.mp4"
        n = _rectify_one(lens, video, rectifier, out_path, args.fourcc)
        print(f"[{lens}] wrote {out_path} ({n} frames)")

    # K is identical across lenses for the same FOV + output size, but we
    # save under both keys for symmetry with the source intrinsics layout.
    any_K = next(iter(rectifiers.values())).K_pinhole
    pinhole_path = args.output_dir / "pinhole_intrinsics.npz"
    np.savez(
        str(pinhole_path),
        K_front=any_K,
        D_front=np.zeros(5, dtype=np.float64),
        K_back=any_K,
        D_back=np.zeros(5, dtype=np.float64),
        image_size=np.array(out_size, dtype=np.int32),
        fov_deg=args.fov_deg,
    )
    print(f"Wrote {pinhole_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
