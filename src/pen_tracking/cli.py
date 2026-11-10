"""Command-line interface for pen tracking."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pen_tracking.config import TrackerConfig
from pen_tracking.paths import DEFAULT_MODEL, DEFAULT_OUTPUT_DIR, DEFAULT_VIDEO
from pen_tracking.video_processor import process_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Markerless hand and pen tracking for navigation. "
            "Detects hand landmarks and tracks the pen tip held in the hand."
        ),
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=DEFAULT_VIDEO,
        help=f"Input video path (default: {DEFAULT_VIDEO})",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"Hand landmarker model path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "tracked_palm1.mp4",
        help="Output annotated video path (use --no-output to skip writing)",
    )
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="Do not write an output video file",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show a live preview window (press q to quit)",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=2,
        help="Process every Nth frame; 2 = ~15 fps, 1 = ~30 fps, 10 = ~3 fps (default: 2)",
    )
    parser.add_argument(
        "--max-hands",
        type=int,
        default=1,
        help="Maximum number of hands to detect (default: 1)",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
        help="Minimum hand detection confidence (default: 0.5)",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.5,
        help="Minimum hand tracking confidence (default: 0.5)",
    )
    parser.add_argument(
        "--pen-max-value",
        type=int,
        default=160,
        help="HSV value threshold for dark pen pixels; raise for out-of-focus blur (default: 160)",
    )
    parser.add_argument(
        "--pen-axis-distance",
        type=float,
        default=20.0,
        help="Max distance from pen axis during detection (default: 20)",
    )
    parser.add_argument(
        "--pen-smoothing",
        type=float,
        default=0.65,
        help="Temporal smoothing factor for pen tip (0-1, default: 0.65)",
    )
    parser.add_argument(
        "--mode",
        choices=("both", "hand", "pen"),
        default="both",
        help="What to track: both (default), hand only, or pen only",
    )
    parser.add_argument(
        "--line-color",
        choices=("yellow", "lime", "cyan", "magenta", "orange", "white", "red"),
        default="yellow",
        help="Color of the pen tracking line (default: yellow)",
    )
    parser.add_argument(
        "--open-output",
        action="store_true",
        help="Open the output video after processing finishes",
    )
    parser.add_argument(
        "--no-pen-without-hand",
        action="store_true",
        help="Only track the pen while a hand is detected (default: also find it by its look)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    config = TrackerConfig(
        video_path=args.video,
        model_path=args.model,
        output_path=None if args.no_output else args.output,
        frame_stride=args.frame_stride,
        max_num_hands=args.max_hands,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        pen_max_value=args.pen_max_value,
        pen_axis_distance=args.pen_axis_distance,
        pen_smoothing=args.pen_smoothing,
        track_mode=args.mode,
        track_pen_without_hand=not args.no_pen_without_hand,
        overlay_color=args.line_color,
        open_output_when_done=args.open_output,
        display=args.display,
    )

    try:
        stats = process_video(config)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        logging.error("%s", exc)
        return 1

    if stats.frames_processed == 0:
        logging.error("No frames were processed.")
        return 1

    if args.open_output and config.output_path is not None and config.output_path.is_file():
        import os
        import subprocess

        path = config.output_path
        if sys.platform == "win32":
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
