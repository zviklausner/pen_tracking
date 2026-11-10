"""Configuration for the pen tracking pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pen_tracking.paths import DEFAULT_MODEL, DEFAULT_OUTPUT_DIR, DEFAULT_VIDEO


@dataclass(frozen=True)
class TrackerConfig:
    """Runtime settings for video hand-occlusion tracking."""

    video_path: Path = DEFAULT_VIDEO
    model_path: Path = DEFAULT_MODEL
    output_path: Path | None = DEFAULT_OUTPUT_DIR / "tracked_palm1.mp4"
    frame_stride: int = 2
    max_num_hands: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    pen_max_value: int = 160
    pen_axis_distance: float = 20.0
    pen_smoothing: float = 0.65
    track_pen_without_hand: bool = True
    display: bool = False
    quit_key: str = "q"

    def validate(self) -> None:
        if not self.video_path.is_file():
            raise FileNotFoundError(f"Video not found: {self.video_path}")
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Hand landmarker model not found: {self.model_path}. "
                "Download it to models/hand_landmarker.task."
            )
        if self.frame_stride <= 0:
            raise ValueError("frame_stride must be positive")
        if self.output_path is not None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
