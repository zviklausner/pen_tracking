"""Draw tracking overlays on video frames."""

from __future__ import annotations

import math

import cv2
import numpy as np
from mediapipe.tasks.python.vision import HandLandmarksConnections

from pen_tracking.config import DEFAULT_OVERLAY_COLOR, DEFAULT_OVERLAY_THICKNESS, OVERLAY_COLORS_BGR
from pen_tracking.tracking_result import FrameTrackingResult

TITLE_BOTH = "AInav Core - Hand + Pen Tracking"
TITLE_HAND = "AInav Core - Hand Tracking"
TITLE_PEN = "AInav Core - Pen Tracking"
STATE_HAND_PEN = "STATE: Hand + Pen tracked"
STATE_HAND_ONLY = "STATE: Hand tracked (pen not found)"
STATE_HAND = "STATE: Hand tracked"
STATE_PEN_ONLY = "STATE: Pen tracked (no hand)"
STATE_PEN = "STATE: Pen tracked"
STATE_CLEAR = "STATE: Clear Line of Sight"

_COLOR_TITLE = (0, 255, 0)
_COLOR_HAND = (0, 255, 0)
_COLOR_CONNECTION = (255, 255, 255)
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


class PenPath:
    """The trail left by the pen tip. A sudden jump starts a new stroke."""

    def __init__(self) -> None:
        self.strokes: list[list[tuple[float, float]]] = []

    def add(self, tip: tuple[float, float], *, frame_height: int) -> None:
        point = (float(tip[0]), float(tip[1]))
        gap = 0.05 * frame_height
        if self.strokes and self.strokes[-1]:
            last = self.strokes[-1][-1]
            jump = math.hypot(point[0] - last[0], point[1] - last[1])
            if jump > gap:
                self.strokes.append([point])
                return
            self.strokes[-1].append(point)
            return
        if not self.strokes:
            self.strokes.append([point])
        else:
            self.strokes[-1].append(point)

    def break_stroke(self) -> None:
        """Do not connect across a frame where the pen was missing."""
        if self.strokes and self.strokes[-1]:
            self.strokes.append([])


def _draw_pen_path(
    frame,
    path: PenPath,
    *,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    outline = max(thickness + 2, thickness * 2)
    for stroke in path.strokes:
        if len(stroke) < 2:
            continue
        points = np.array([(int(x), int(y)) for x, y in stroke], dtype=np.int32)
        cv2.polylines(frame, [points], False, (0, 0, 0), outline, cv2.LINE_AA)
        cv2.polylines(frame, [points], False, color, thickness, cv2.LINE_AA)


def _draw_pen(
    frame,
    pen,
    *,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    grip = (int(pen.grip[0]), int(pen.grip[1]))
    tip = (int(pen.tip[0]), int(pen.tip[1]))
    # Dark outline so a bright line stays visible on both wood and black shaft.
    outline = max(thickness + 2, thickness * 2)
    cv2.line(frame, grip, tip, (0, 0, 0), outline)
    cv2.line(frame, grip, tip, color, thickness)
    radius = max(6, thickness + 2)
    cv2.circle(frame, grip, radius, (0, 0, 0), -1)
    cv2.circle(frame, grip, max(4, thickness), color, -1)
    cv2.circle(frame, tip, radius + 2, (0, 0, 0), -1)
    cv2.circle(frame, tip, radius, color, -1)
    cv2.putText(
        frame,
        "PEN TIP",
        (tip[0] + 10, tip[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        3,
    )
    cv2.putText(
        frame,
        "PEN TIP",
        (tip[0] + 10, tip[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
    )


def _draw_top_right_badge(frame, lines: list[str]) -> None:
    """Draw a readable multi-line badge in the top-right corner."""
    if not lines:
        return
    height, width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.15
    thickness = 3
    sizes = [cv2.getTextSize(line, font, scale, thickness) for line in lines]
    text_w = max(size[0][0] for size in sizes)
    text_h = sizes[0][0][1]
    baseline = sizes[0][1]
    line_gap = 10
    pad_x, pad_y = 20, 16
    box_w = text_w + pad_x * 2
    box_h = pad_y * 2 + len(lines) * text_h + (len(lines) - 1) * line_gap + baseline
    x1 = max(8, width - box_w - 16)
    y1 = 14
    x2 = x1 + box_w
    y2 = y1 + box_h
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
    y = y1 + pad_y + text_h
    for line in lines:
        cv2.putText(
            frame,
            line,
            (x1 + pad_x, y),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        y += text_h + line_gap


def _draw_frame_badge(frame, frame_index: int) -> None:
    """Draw frame index in the top-right corner."""
    _draw_top_right_badge(frame, [f"FRAME {frame_index}"])


def render_frame(
    frame,
    tracking: FrameTrackingResult,
    *,
    draw_hand: bool = True,
    draw_pen: bool = True,
    draw_pen_line: bool = True,
    draw_pen_path: bool = False,
    pen_path: PenPath | None = None,
    overlay_color: tuple[int, int, int] | str = DEFAULT_OVERLAY_COLOR,
    overlay_thickness: int = DEFAULT_OVERLAY_THICKNESS,
    frame_index: int | None = None,
) -> None:
    """Annotate a BGR frame in place with hand and/or pen tracking overlays."""
    if isinstance(overlay_color, str):
        color = OVERLAY_COLORS_BGR[overlay_color]
    else:
        color = overlay_color

    if draw_hand and draw_pen:
        title = TITLE_BOTH
    elif draw_hand:
        title = TITLE_HAND
    else:
        title = TITLE_PEN
    cv2.putText(
        frame,
        title,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        _COLOR_TITLE,
        2,
    )
    if frame_index is not None:
        _draw_frame_badge(frame, frame_index)

    show_hand = draw_hand and tracking.has_hand
    show_pen = draw_pen and tracking.has_pen

    if show_hand and show_pen:
        status, status_color = STATE_HAND_PEN, _COLOR_STATUS_BOTH
    elif show_hand and draw_pen:
        status, status_color = STATE_HAND_ONLY, _COLOR_STATUS_HAND
    elif show_hand:
        status, status_color = STATE_HAND, _COLOR_STATUS_HAND
    elif show_pen and draw_hand:
        status, status_color = STATE_PEN_ONLY, _COLOR_STATUS_BOTH
    elif show_pen:
        status, status_color = STATE_PEN, _COLOR_STATUS_BOTH
    else:
        status, status_color = STATE_CLEAR, _COLOR_STATUS_CLEAR

    cv2.putText(
        frame,
        status,
        (15, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        status_color,
        2,
    )

    if show_hand:
        height, width = frame.shape[:2]
        for hand in tracking.hand.hand_landmarks:
            _draw_landmarks(frame, hand, width, height)

    if draw_pen_path and pen_path is not None:
        _draw_pen_path(frame, pen_path, color=color, thickness=overlay_thickness)

    if show_pen and draw_pen_line and tracking.pen is not None:
        _draw_pen(
            frame,
            tracking.pen,
            color=color,
            thickness=overlay_thickness,
        )


def resize_frame(frame, target_height: int):
    """Downsample a frame to target height while preserving aspect ratio."""
    height, width = frame.shape[:2]
    scale = target_height / height
    target_width = int(width * scale)
    return cv2.resize(frame, (target_width, target_height))
