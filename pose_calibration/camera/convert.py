#!/usr/bin/env python3
"""Convert an Insta360 ``.insv`` recording into per-lens ``.mp4`` files.

X4 / X5 cameras store the two lenses as **separate video streams** in
one ``.insv`` container, not side-by-side in a single dual-fisheye
frame. ``cv2.VideoCapture`` only reads stream 0, so we must demux the
two tracks up-front with ffmpeg. The video tracks are standard H.264,
so this is a stream-copy — no re-encode::

    pixi run python -m pose_calibration.camera.convert \\
        --input data/VID_20260515_120000_00_001.insv

Output (next to the input, by default)::

    <stem>_lens0.mp4
    <stem>_lens1.mp4

Stream 0 is typically the screen-side ("front") lens on the X4/X5, but
**verify with a one-frame inspection** before labelling — wiring the
wrong lens into pose code will silently produce mirrored / 180°-rotated
poses. The IMU track lives in an Insta360-proprietary box and is **not**
carried across.
"""

from __future__ import annotations

import dataclasses
import shutil
import subprocess
import sys
from pathlib import Path  # noqa: TC003 — tyro needs Path at runtime

import tyro


@dataclasses.dataclass
class Args:
    """Demux an .insv into two per-lens .mp4 streams (ffmpeg stream-copy)."""

    input: Path
    """Source .insv file."""

    output_lens0: Path | None = None
    """Output for stream 0. Defaults to <input>_lens0.mp4."""

    output_lens1: Path | None = None
    """Output for stream 1. Defaults to <input>_lens1.mp4."""

    force: bool = False
    """Overwrite any existing outputs."""


def _extract_stream(input_path: Path, output_path: Path, stream_index: int, force: bool) -> Path:
    if output_path.exists() and not force:
        print(f"{output_path} exists; pass --force to overwrite. Skipping.")
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y" if force else "-n",
        "-i",
        str(input_path),
        "-map",
        f"0:v:{stream_index}",
        "-c:v",
        "copy",
        "-an",
        str(output_path),
    ]
    print(f"$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}")
    print(f"Wrote {output_path}")
    return output_path


def convert(
    input_path: Path,
    output_lens0: Path,
    output_lens1: Path,
    force: bool = False,
) -> tuple[Path, Path]:
    """Demux the two video streams of ``input_path`` into separate mp4 files."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on PATH (pixi.toml includes it; run inside `pixi shell`)"
        )
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    _extract_stream(input_path, output_lens0, 0, force)
    _extract_stream(input_path, output_lens1, 1, force)
    return output_lens0, output_lens1


def main(args: Args) -> None:
    out0 = args.output_lens0 or args.input.with_name(args.input.stem + "_lens0.mp4")
    out1 = args.output_lens1 or args.input.with_name(args.input.stem + "_lens1.mp4")
    try:
        convert(args.input, out0, out1, force=args.force)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main(tyro.cli(Args))
