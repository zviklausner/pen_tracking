"""Combined hand and pen tracking result."""

from __future__ import annotations

from dataclasses import dataclass

from pen_tracking.hand_tracker import HandDetectionResult
from pen_tracking.pen_tracker_cv import PenDetectionResult


@dataclass(frozen=True)
class FrameTrackingResult:
    """Per-frame hand and pen tracking output."""

    hand: HandDetectionResult
    pen: PenDetectionResult | None = None

    @property
    def has_hand(self) -> bool:
        return bool(self.hand.hand_landmarks)

    @property
    def has_pen(self) -> bool:
        return self.pen is not None

    @property
    def is_occluded(self) -> bool:
        return self.has_hand
