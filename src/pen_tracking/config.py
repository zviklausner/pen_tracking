"""Configuration for the pen tracking pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pen_tracking.paths import DEFAULT_MODEL, DEFAULT_OUTPUT_DIR, DEFAULT_VIDEO

TrackMode = Literal["both", "hand", "pen"]
TRACK_MODES: tuple[TrackMode, ...] = ("both", "hand", "pen")

# OpenCV BGR colors that stand out against a dark pencil.
OverlayColorName = Literal[
    "yellow", "lime", "cyan", "magenta", "orange", "white", "red"
]
OVERLAY_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "yellow": (0, 255, 255),
    "lime": (0, 255, 0),
    "cyan": (255, 255, 0),
    "magenta": (255, 0, 255),
    "orange": (0, 165, 255),
    "white": (255, 255, 255),
    "red": (0, 0, 255),
}
DEFAULT_OVERLAY_COLOR: OverlayColorName = "yellow"
DEFAULT_OVERLAY_THICKNESS = 5


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
    track_mode: TrackMode = "both"
    track_pen_without_hand: bool = True
    overlay_color: OverlayColorName = DEFAULT_OVERLAY_COLOR
    overlay_thickness: int = DEFAULT_OVERLAY_THICKNESS
    open_output_when_done: bool = False
    display: bool = False
    quit_key: str = "q"

    @property
    def track_hand(self) -> bool:
        return self.track_mode in ("both", "hand")

    @property
    def track_pen(self) -> bool:
        return self.track_mode in ("both", "pen")

    def validate(self) -> None:
        if not self.video_path.is_file():
            raise FileNotFoundError(f"Video not found: {self.video_path}")
        if self.track_mode not in TRACK_MODES:
            raise ValueError(f"track_mode must be one of {TRACK_MODES}")
        if self.track_hand and not self.model_path.is_file():
            raise FileNotFoundError(
                f"Hand landmarker model not found: {self.model_path}. "
                "Download it to models/hand_landmarker.task."
            )
        if self.frame_stride <= 0:
            raise ValueError("frame_stride must be positive")
        if self.overlay_color not in OVERLAY_COLORS_BGR:
            raise ValueError(f"overlay_color must be one of {tuple(OVERLAY_COLORS_BGR)}")
        if self.overlay_thickness <= 0:
            raise ValueError("overlay_thickness must be positive")
        if self.output_path is not None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def overlay_bgr(self) -> tuple[int, int, int]:
        return OVERLAY_COLORS_BGR[self.overlay_color]
