"""MediaPipe hand landmark detection wrapper."""

from __future__ import annotations

from dataclasses import dataclass

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision


@dataclass(frozen=True)
class HandDetectionResult:
    """Per-frame hand detection output."""

    hand_landmarks: list[list[vision.NormalizedLandmark]]
    handedness: list[str]


class HandTracker:
    """Detects hand landmarks using the MediaPipe Tasks HandLandmarker API."""

    def __init__(
        self,
        model_path: str,
        *,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, rgb_frame, timestamp_ms: int) -> HandDetectionResult:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        landmarks: list[list] = []
        handedness_labels: list[str] = []

        if result.hand_landmarks:
            landmarks = [list(hand) for hand in result.hand_landmarks]
        if result.handedness:
            for categories in result.handedness:
                if categories:
                    handedness_labels.append(categories[0].category_name)

        return HandDetectionResult(
            hand_landmarks=landmarks,
            handedness=handedness_labels,
        )

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> HandTracker:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
