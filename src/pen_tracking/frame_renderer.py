"""Draw tracking overlays on video frames."""

from __future__ import annotations

import cv2
from mediapipe.tasks.python.vision import HandLandmarksConnections

from pen_tracking.hand_tracker import HandDetectionResult

TITLE = "AInav Core - Markerless Tracking"
STATE_OCCLUSION = "STATE: Occlusion (Hand Mask Active)"
STATE_CLEAR = "STATE: Clear Line of Sight"

_COLOR_TITLE = (0, 255, 0)
_COLOR_OCCLUSION = (0, 0, 255)
_COLOR_CLEAR = (0, 255, 255)
_COLOR_LANDMARK = (0, 255, 0)
_COLOR_CONNECTION = (255, 255, 255)


def _draw_landmarks(frame, hand_landmarks, width: int, height: int) -> None:
    for landmark in hand_landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        cv2.circle(frame, (x, y), 3, _COLOR_LANDMARK, -1)

    for connection in HandLandmarksConnections.HAND_CONNECTIONS:
        start = hand_landmarks[connection.start]
        end = hand_landmarks[connection.end]
        start_point = (int(start.x * width), int(start.y * height))
        end_point = (int(end.x * width), int(end.y * height))
        cv2.line(frame, start_point, end_point, _COLOR_CONNECTION, 2)


def render_frame(frame, detection: HandDetectionResult) -> None:
    """Annotate a BGR frame in place with status text and hand landmarks."""
    cv2.putText(
        frame,
        TITLE,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        _COLOR_TITLE,
        2,
    )

    if detection.hand_landmarks:
        cv2.putText(
            frame,
            STATE_OCCLUSION,
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            _COLOR_OCCLUSION,
            2,
        )
        height, width = frame.shape[:2]
        for hand in detection.hand_landmarks:
            _draw_landmarks(frame, hand, width, height)
    else:
        cv2.putText(
            frame,
            STATE_CLEAR,
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            _COLOR_CLEAR,
            2,
        )


def resize_frame(frame, target_height: int):
    """Downsample a frame to target height while preserving aspect ratio."""
    height, width = frame.shape[:2]
    scale = target_height / height
    target_width = int(width * scale)
    return cv2.resize(frame, (target_width, target_height))
