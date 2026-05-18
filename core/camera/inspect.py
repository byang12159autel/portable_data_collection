#!/usr/bin/env python3
"""Quick contents check for an Insta360 ``.insv`` (or any video container).

Runs ``ffprobe`` to list the video / audio streams, then dumps the first
frame of each video stream as a PNG so you can eyeball the layout — most
useful for confirming which stream is which lens before running
calibration::

    pixi run python -m core.camera.inspect \\
        --input data/VID_20260515_120000_00_001.insv

By default the PNGs land next to the input as
``<stem>_stream<n>_frame0.png``.
"""

from __future__ import annotations

import dataclasses
import shutil
import subprocess
import sys
from pathlib import Path  # noqa: TC003 — tyro needs Path at runtime

import cv2
import tyro


@dataclasses.dataclass
class Args:
    """Inspect an .insv / .mp4 container."""

    input: Path
    """Source video file."""

    skip_dump: bool = False
    """Skip the per-stream frame dump — just print stream info."""

    output_dir: Path | None = None
    """Where to write the per-stream PNGs. Defaults to the input's directory."""


def _run_ffprobe(input_path: Path) -> int:
    """Print stream info via ffprobe. Returns the number of video streams."""
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe not found on PATH (pixi env includes it)")

    print(f"\n=== ffprobe: {input_path} ===")
    info = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries",
            "stream=index,codec_type,codec_name,width,height,r_frame_rate,duration",
            "-of", "default=noprint_wrappers=1",
            str(input_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if info.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {info.stderr.strip()}")
    print(info.stdout.strip())

    # Count video streams for the dump loop.
    count = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v",
            "-show_entries", "stream=index",
            "-of", "csv=p=0",
            str(input_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    n_video = len([x for x in count.stdout.splitlines() if x.strip()])
    return n_video


def _dump_first_frame_per_stream(
    input_path: Path, n_streams: int, out_dir: Path,
) -> None:
    """Use ffmpeg + cv2 to write one PNG per video stream."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n=== first-frame dump ===")
    for i in range(n_streams):
        # Use ffmpeg to extract a single frame from stream i to a temp mp4
        # (smaller than re-implementing demux), then read it with cv2.
        # Stream-copy a tiny segment first to avoid re-encoding the whole file.
        tmp = out_dir / f".{input_path.stem}_stream{i}_tmp.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-map", f"0:v:{i}",
            "-c:v", "copy",
            "-frames:v", "1",
            str(tmp),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            print(f"  stream {i}: ffmpeg failed ({proc.stderr.strip().splitlines()[-1]})")
            continue

        cap = cv2.VideoCapture(str(tmp))
        ok, bgr = cap.read()
        cap.release()
        tmp.unlink(missing_ok=True)
        if not ok:
            print(f"  stream {i}: cv2 failed to decode")
            continue
        out_path = out_dir / f"{input_path.stem}_stream{i}_frame0.png"
        cv2.imwrite(str(out_path), bgr)
        print(f"  stream {i}: {bgr.shape[1]}x{bgr.shape[0]} -> {out_path}")


def main(args: Args) -> None:
    if not args.input.is_file():
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    try:
        n_video = _run_ffprobe(args.input)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not args.skip_dump:
        out_dir = args.output_dir or args.input.parent
        _dump_first_frame_per_stream(args.input, n_video, out_dir)


if __name__ == "__main__":
    main(tyro.cli(Args))
