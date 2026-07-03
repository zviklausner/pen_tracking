"""Cross-platform video I/O helpers."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2


def create_video_writer(
    output_path: Path,
    fps: float,
    width: int,
    height: int,
) -> cv2.VideoWriter:
    """Create a VideoWriter, trying codecs that work on macOS, Linux, and Windows."""
    suffix = output_path.suffix.lower()
    if suffix in {".mp4", ".m4v", ".mov"}:
        codecs = ("mp4v", "avc1", "H264")
    elif suffix == ".avi":
        codecs = ("XVID", "MJPG")
    else:
        codecs = ("mp4v",)

    for codec in codecs:
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
        )
        if writer.isOpened():
            return writer

    raise RuntimeError(
        f"Unable to create output video: {output_path}. "
        "Try a different file extension such as .avi."
    )


def supports_gui() -> bool:
    """Return False when running headless (no display available)."""
    if sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        return False
    return True
