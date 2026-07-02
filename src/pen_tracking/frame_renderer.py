"""Draw tracking overlays on video frames."""

from __future__ import annotations

import cv2
from mediapipe.tasks.python.vision import HandLandmarksConnections

from pen_tracking.tracking_result import FrameTrackingResult

TITLE = "AInav Core - Hand + Pen Tracking"
STATE_HAND_PEN = "STATE: Hand + Pen tracked"
STATE_HAND_ONLY = "STATE: Hand tracked (pen not found)"
STATE_CLEAR = "STATE: Clear Line of Sight"

_COLOR_TITLE = (0, 255, 0)
_COLOR_HAND = (0, 255, 0)
_COLOR_CONNECTION = (255, 255, 255)
_COLOR_PEN_BODY = (255, 128, 0)
_COLOR_PEN_TIP = (0, 0, 255)
_COLOR_PEN_GRIP = (255, 0, 255)
_COLOR_STATUS_HAND = (0, 0, 255)
_COLOR_STATUS_BOTH = (0, 255, 255)
_COLOR_STATUS_CLEAR = (0, 255, 255)


def _draw_landmarks(frame, hand_landmarks, width: int, height: int) -> None:
    for landmark in hand_landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        cv2.circle(frame, (x, y), 3, _COLOR_HAND, -1)

    for connection in HandLandmarksConnections.HAND_CONNECTIONS:
        start = hand_landmarks[connection.start]
        end = hand_landmarks[connection.end]
        start_point = (int(start.x * width), int(start.y * height))
        end_point = (int(end.x * width), int(end.y * height))
        cv2.line(frame, start_point, end_point, _COLOR_CONNECTION, 2)


def _draw_pen(frame, pen) -> None:
    grip = (int(pen.grip[0]), int(pen.grip[1]))
    tip = (int(pen.tip[0]), int(pen.tip[1]))
    cv2.line(frame, grip, tip, _COLOR_PEN_BODY, 2)
    cv2.circle(frame, grip, 5, _COLOR_PEN_GRIP, -1)
    cv2.circle(frame, tip, 7, _COLOR_PEN_TIP, -1)
    cv2.putText(
        frame,
        "PEN TIP",
        (tip[0] + 8, tip[1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        _COLOR_PEN_TIP,
        1,
    )


def render_frame(frame, tracking: FrameTrackingResult) -> None:
    """Annotate a BGR frame in place with hand and pen tracking overlays."""
    cv2.putText(
        frame,
        TITLE,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        _COLOR_TITLE,
        2,
    )

    if tracking.has_hand:
        if tracking.has_pen:
            status = STATE_HAND_PEN
            color = _COLOR_STATUS_BOTH
        else:
            status = STATE_HAND_ONLY
            color = _COLOR_STATUS_HAND

        cv2.putText(
            frame,
            status,
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

        height, width = frame.shape[:2]
        for hand in tracking.hand.hand_landmarks:
            _draw_landmarks(frame, hand, width, height)

        if tracking.pen is not None:
            _draw_pen(frame, tracking.pen)
    else:
        cv2.putText(
            frame,
            STATE_CLEAR,
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            _COLOR_STATUS_CLEAR,
            2,
        )


def resize_frame(frame, target_height: int):
    """Downsample a frame to target height while preserving aspect ratio."""
    height, width = frame.shape[:2]
    scale = target_height / height
    target_width = int(width * scale)
    return cv2.resize(frame, (target_width, target_height))
